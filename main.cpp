#include "httplib.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <set>
#include <sstream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <climits>

static const int DIMS = 60;   // demo vectors
// Doc embeddings dimension is determined at runtime from Ollama's model output

// =====================================================================
//  DATA TYPES
// =====================================================================

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

// =====================================================================
//  DISTANCE METRICS
// =====================================================================

float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (int i = 0; i < (int)a.size(); i++) { float d = a[i]-b[i]; s += d*d; }
    return std::sqrt(s);
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    float dot=0, na=0, nb=0;
    for (int i = 0; i < (int)a.size(); i++) {
        dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (int i = 0; i < (int)a.size(); i++) s += std::abs(a[i]-b[i]);
    return s;
}

DistFn getDistFn(const std::string& m) {
    if (m == "cosine")    return cosine;
    if (m == "manhattan") return manhattan;
    return euclidean;
}

// =====================================================================
//  BRUTE FORCE
// =====================================================================

class BruteForce {
public:
    std::vector<VectorItem> items;

    void insert(const VectorItem& v) { items.push_back(v); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::vector<std::pair<float,int>> r;
        r.reserve(items.size());
        for (auto& v : items) r.push_back({dist(q, v.emb), v.id});
        std::sort(r.begin(), r.end());
        if ((int)r.size() > k) r.resize(k);
        return r;
    }

    void remove(int id) {
        items.erase(std::remove_if(items.begin(), items.end(),
            [id](const VectorItem& v){ return v.id == id; }), items.end());
    }
};

// =====================================================================
//  KD-TREE
// =====================================================================

struct KDNode {
    VectorItem item;
    KDNode* left  = nullptr;
    KDNode* right = nullptr;
    explicit KDNode(const VectorItem& v) : item(v) {}
};

class KDTree {
    KDNode* root = nullptr;
    int dims;

    void destroy(KDNode* n) {
        if (!n) return; destroy(n->left); destroy(n->right); delete n;
    }

    KDNode* ins(KDNode* n, const VectorItem& v, int d) {
        if (!n) return new KDNode(v);
        int ax = d % dims;
        if (v.emb[ax] < n->item.emb[ax]) n->left  = ins(n->left,  v, d+1);
        else                              n->right = ins(n->right, v, d+1);
        return n;
    }

    void knn(KDNode* n, const std::vector<float>& q, int k, int d, DistFn dist,
             std::priority_queue<std::pair<float,int>>& heap)
    {
        if (!n) return;
        float dn = dist(q, n->item.emb);
        if ((int)heap.size() < k || dn < heap.top().first) {
            heap.push({dn, n->item.id});
            if ((int)heap.size() > k) heap.pop();
        }
        int ax = d % dims;
        float diff = q[ax] - n->item.emb[ax];
        KDNode* closer  = diff < 0 ? n->left  : n->right;
        KDNode* farther = diff < 0 ? n->right : n->left;
        knn(closer, q, k, d+1, dist, heap);
        if ((int)heap.size() < k || std::abs(diff) < heap.top().first)
            knn(farther, q, k, d+1, dist, heap);
    }

public:
    explicit KDTree(int d) : dims(d) {}
    ~KDTree() { destroy(root); }

    void insert(const VectorItem& v) { root = ins(root, v, 0); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::priority_queue<std::pair<float,int>> heap;
        knn(root, q, k, 0, dist, heap);
        std::vector<std::pair<float,int>> r;
        while (!heap.empty()) { r.push_back(heap.top()); heap.pop(); }
        std::sort(r.begin(), r.end());
        return r;
    }

    void rebuild(const std::vector<VectorItem>& items) {
        destroy(root); root = nullptr;
        for (auto& v : items) insert(v);
    }
};

// =====================================================================
//  HNSW — Hierarchical Navigable Small World
// =====================================================================

class HNSW {
    struct Node {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> G;
    int    M, M0, ef_build;
    float  mL;
    int    topLayer = -1;
    int    entryPt  = -1;
    std::mt19937 rng;

    int randLevel() {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        return (int)std::floor(-std::log(u(rng)) * mL);
    }

    std::vector<std::pair<float,int>> searchLayer(
        const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist)
    {
        std::unordered_map<int,bool> vis;
        std::priority_queue<std::pair<float,int>,
            std::vector<std::pair<float,int>>, std::greater<>> cands;
        std::priority_queue<std::pair<float,int>> found;

        float d0 = dist(q, G[ep].item.emb);
        vis[ep] = true;
        cands.push({d0, ep});
        found.push({d0, ep});

        while (!cands.empty()) {
            auto [cd, cid] = cands.top(); cands.pop();
            if ((int)found.size() >= ef && cd > found.top().first) break;
            if (lyr >= (int)G[cid].nbrs.size()) continue;
            for (int nid : G[cid].nbrs[lyr]) {
                if (vis[nid] || !G.count(nid)) continue;
                vis[nid] = true;
                float nd = dist(q, G[nid].item.emb);
                if ((int)found.size() < ef || nd < found.top().first) {
                    cands.push({nd, nid});
                    found.push({nd, nid});
                    if ((int)found.size() > ef) found.pop();
                }
            }
        }

        std::vector<std::pair<float,int>> res;
        while (!found.empty()) { res.push_back(found.top()); found.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    std::vector<int> selectNbrs(std::vector<std::pair<float,int>>& cands, int maxM) {
        std::vector<int> r;
        for (int i = 0; i < std::min((int)cands.size(), maxM); i++)
            r.push_back(cands[i].second);
        return r;
    }

public:
    HNSW(int m = 16, int efBuild = 200)
        : M(m), M0(2*m), ef_build(efBuild),
          mL(1.0f / std::log((float)m)), rng(42) {}

    void insert(const VectorItem& item, DistFn dist) {
        int id  = item.id;
        int lvl = randLevel();
        G[id]   = {item, lvl, std::vector<std::vector<int>>(lvl + 1)};

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        for (int lc = std::min(topLayer, lvl); lc >= 0; lc--) {
            auto W   = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            auto sel = selectNbrs(W, maxM);
            G[id].nbrs[lc] = sel;

            for (int nid : sel) {
                if (!G.count(nid)) continue;
                if ((int)G[nid].nbrs.size() <= lc) G[nid].nbrs.resize(lc + 1);
                auto& conn = G[nid].nbrs[lc];
                conn.push_back(id);
                if ((int)conn.size() > maxM) {
                    std::vector<std::pair<float,int>> ds;
                    for (int c : conn) if (G.count(c))
                        ds.push_back({dist(G[nid].item.emb, G[c].item.emb), c});
                    std::sort(ds.begin(), ds.end());
                    conn.clear();
                    for (int i = 0; i < maxM && i < (int)ds.size(); i++)
                        conn.push_back(ds[i].second);
                }
            }
            if (!W.empty()) ep = W[0].second;
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, int ef, DistFn dist)
    {
        if (entryPt == -1) return {};
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(q, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        auto W = searchLayer(q, ep, std::max(ef, k), 0, dist);
        if ((int)W.size() > k) W.resize(k);
        return W;
    }

    void remove(int id) {
        if (!G.count(id)) return;
        for (auto& [nid, nd] : G)
            for (auto& layer : nd.nbrs)
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());
        if (entryPt == id) {
            entryPt = -1;
            for (auto& [nid, nd] : G) if (nid != id) { entryPt = nid; break; }
        }
        G.erase(id);
    }

    struct GraphInfo {
        int topLayer, nodeCount;
        std::vector<int> nodesPerLayer, edgesPerLayer;
        struct NV { int id; std::string metadata, category; int maxLyr; };
        struct EV { int src, dst, lyr; };
        std::vector<NV> nodes;
        std::vector<EV> edges;
    };

    GraphInfo getInfo() {
        GraphInfo gi;
        gi.topLayer  = topLayer;
        gi.nodeCount = (int)G.size();
        int maxL = std::max(topLayer + 1, 1);
        gi.nodesPerLayer.assign(maxL, 0);
        gi.edgesPerLayer.assign(maxL, 0);
        for (auto& [id, nd] : G) {
            gi.nodes.push_back({id, nd.item.metadata, nd.item.category, nd.maxLyr});
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < (int)nd.nbrs.size())
                    for (int nid : nd.nbrs[lc])
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.push_back({id, nid, lc});
                        }
            }
        }
        return gi;
    }

    size_t size() const { return G.size(); }
};

// =====================================================================
//  VECTOR DATABASE  (demo 60D index)
// =====================================================================

class VectorDB {
    std::unordered_map<int, VectorItem> store;
    BruteForce bf;
    KDTree     kdt;
    HNSW       hnsw;
    std::mutex mu;
    int nextId = 1;

public:
    const int dims;
    explicit VectorDB(int d) : kdt(d), hnsw(16, 200), dims(d) {}

    int insert(const std::string& meta, const std::string& cat,
               const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        VectorItem v{nextId++, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
        return v.id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); bf.remove(id); hnsw.remove(id);
        std::vector<VectorItem> rem;
        for (auto& [i, v] : store) rem.push_back(v);
        kdt.rebuild(rem);
        return true;
    }

    struct Hit { int id; std::string meta, cat; std::vector<float> emb; float dist; };
    struct SearchOut { std::vector<Hit> hits; long long us; std::string algo, metric; };

    SearchOut search(const std::vector<float>& q, int k,
                     const std::string& metric, const std::string& algo)
    {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn = getDistFn(metric);
        auto t0  = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<float,int>> raw;
        if      (algo == "bruteforce") raw = bf.knn(q, k, dfn);
        else if (algo == "kdtree")     raw = kdt.knn(q, k, dfn);
        else                           raw = hnsw.knn(q, k, 50, dfn);

        long long us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        SearchOut out; out.us = us; out.algo = algo; out.metric = metric;
        for (auto& [d, id] : raw)
            if (store.count(id))
                out.hits.push_back({id, store[id].metadata, store[id].category, store[id].emb, d});
        return out;
    }

    struct BenchOut { long long bfUs, kdUs, hnswUs; int n; };

    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn  = getDistFn(metric);
        auto time = [&](auto fn) -> long long {
            auto t = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t).count();
        };
        return {
            time([&]{ bf.knn(q, k, dfn); }),
            time([&]{ kdt.knn(q, k, dfn); }),
            time([&]{ hnsw.knn(q, k, 50, dfn); }),
            (int)store.size()
        };
    }

    std::vector<VectorItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<VectorItem> r;
        for (auto& [id, v] : store) r.push_back(v);
        return r;
    }

    HNSW::GraphInfo hnswInfo() {
        std::lock_guard<std::mutex> lk(mu);
        return hnsw.getInfo();
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }
};

// =====================================================================
//  JSON HELPERS
// =====================================================================

std::string jS(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else                o += c;
    }
    return o + '"';
}

std::string jVec(const std::vector<float>& v) {
    std::ostringstream ss; ss << '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) ss << ',';
        ss << std::fixed << std::setprecision(4) << v[i];
    }
    return ss.str() + ']';
}

std::vector<float> parseVec(const std::string& s) {
    std::vector<float> v;
    std::istringstream ss(s); std::string t;
    while (std::getline(ss, t, ','))
        try { v.push_back(std::stof(t)); } catch (...) {}
    return v;
}

// Extract a JSON string field value (handles basic escape sequences)
std::string extractStr(const std::string& body, const std::string& key) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return "";
    p = body.find(':', p) + 1;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
    if (p >= body.size() || body[p] != '"') return "";
    p++;
    std::string result;
    while (p < body.size()) {
        if (body[p] == '"') break;
        if (body[p] == '\\' && p + 1 < body.size()) {
            p++;
            switch (body[p]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += body[p]; break;
            }
        } else {
            result += body[p];
        }
        p++;
    }
    return result;
}

// Extract a JSON integer field value
int extractInt(const std::string& body, const std::string& key, int def = 0) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return def;
    p = body.find(':', p) + 1;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
    try { return std::stoi(body.substr(p)); } catch (...) { return def; }
}

bool parseBody(const std::string& b, std::string& meta,
               std::string& cat, std::vector<float>& emb)
{
    meta = extractStr(b, "metadata");
    cat  = extractStr(b, "category");
    auto extractArr = [&](const std::string& key) -> std::vector<float> {
        size_t p = b.find('"' + key + '"');
        if (p == std::string::npos) return {};
        p = b.find('[', p);
        if (p == std::string::npos) return {};
        size_t e = b.find(']', p);
        if (e == std::string::npos) return {};
        return parseVec(b.substr(p + 1, e - p - 1));
    };
    emb = extractArr("embedding");
    return !meta.empty() && !emb.empty();
}

struct ChatMessage {
    std::string role;
    std::string content;
};

// Parses sliding-window chat history array: [{"role":"user","content":"..."}, ...]
std::vector<ChatMessage> parseChatHistory(const std::string& body) {
    std::vector<ChatMessage> history;
    size_t p = body.find("\"history\"");
    if (p == std::string::npos) return history;
    p = body.find('[', p);
    if (p == std::string::npos) return history;

    while (p < body.size()) {
        size_t nextRole = body.find("\"role\"", p);
        if (nextRole == std::string::npos) break;
        size_t objStart = body.rfind('{', nextRole);
        if (objStart == std::string::npos) break;

        size_t objEnd = std::string::npos;
        bool inStr = false;
        for (size_t i = objStart; i < body.size(); i++) {
            if (body[i] == '\\' && i + 1 < body.size()) {
                i++; // skip escaped character
                continue;
            }
            if (body[i] == '"') {
                inStr = !inStr;
            } else if (!inStr && body[i] == '}') {
                objEnd = i;
                break;
            }
        }
        if (objEnd == std::string::npos) break;

        std::string obj = body.substr(objStart, objEnd - objStart + 1);
        std::string role = extractStr(obj, "role");
        std::string content = extractStr(obj, "content");
        if (!role.empty() && !content.empty()) {
            history.push_back({role, content});
        }
        p = objEnd + 1;
    }

    // Enforce sliding context limit: retain at most the last 20 messages
    if (history.size() > 20) {
        history.erase(history.begin(), history.begin() + (history.size() - 20));
    }
    return history;
}

void cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

// =====================================================================
//  TEXT CHUNKER
// =====================================================================

std::vector<std::string> chunkText(const std::string& text,
                                   int chunkWords = 250, int overlapWords = 30)
{
    std::istringstream ss(text);
    std::vector<std::string> words;
    std::string w;
    while (ss >> w) words.push_back(w);

    if (words.empty()) return {};
    if ((int)words.size() <= chunkWords) return {text};

    std::vector<std::string> chunks;
    int step = chunkWords - overlapWords;
    for (int i = 0; i < (int)words.size(); i += step) {
        int end = std::min(i + chunkWords, (int)words.size());
        std::string chunk;
        for (int j = i; j < end; j++) { if (j > i) chunk += ' '; chunk += words[j]; }
        chunks.push_back(chunk);
        if (end == (int)words.size()) break;
    }
    return chunks;
}

// =====================================================================
//  OLLAMA CLIENT  — wraps local Ollama REST API
//  Install:  https://ollama.com
//  Models:   ollama pull nomic-embed-text
//            ollama pull llama3.2
// =====================================================================

class OllamaClient {
    std::string host;
    int         port;

    // Escape a string for embedding inside a JSON string literal
    std::string esc(const std::string& s) {
        std::string o;
        for (char c : s) {
            if      (c == '"')  o += "\\\"";
            else if (c == '\\') o += "\\\\";
            else if (c == '\n') o += "\\n";
            else if (c == '\r') o += "\\r";
            else if (c == '\t') o += "\\t";
            else                o += c;
        }
        return o;
    }

    // Parse {"embedding":[...]} from Ollama /api/embeddings response
    std::vector<float> parseEmbedding(const std::string& body) {
        size_t p = body.find("\"embedding\"");
        if (p == std::string::npos) return {};
        p = body.find('[', p);
        if (p == std::string::npos) return {};
        // Find matching ]  — embeddings can be large (768+ floats)
        size_t e = p + 1, depth = 1;
        while (e < body.size() && depth > 0) {
            if (body[e] == '[') depth++;
            else if (body[e] == ']') depth--;
            e++;
        }
        return parseVec(body.substr(p + 1, e - p - 2));
    }

    // Parse {"response":"..."} from Ollama /api/generate response
    std::string parseResponse(const std::string& body) {
        return extractStr(body, "response");
    }

public:
    std::string embedModel = "nomic-embed-text";
    std::string genModel   = "llama3.2";

    OllamaClient(const std::string& h = "127.0.0.1", int p = 11434)
        : host(h), port(p) {}

    bool isAvailable() {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(2, 0);
        auto res = cli.Get("/api/tags");
        return res && res->status == 200;
    }

    // Returns empty vector if Ollama is not running or model not found
    std::vector<float> embed(const std::string& text) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(30, 0);
        std::string body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
        auto res = cli.Post("/api/embeddings", body, "application/json");
        if (!res || res->status != 200) return {};
        return parseEmbedding(res->body);
    }

    // Returns error string if Ollama is unavailable
    std::string generate(const std::string& prompt) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(180, 0);   // LLMs can be slow
        std::string body = "{\"model\":\"" + genModel + "\","
                           "\"prompt\":\"" + esc(prompt) + "\","
                           "\"stream\":false,"
                           "\"options\":{\"stop\":[\"\\nUser:\",\"\\nHuman:\"]}}";
        auto res = cli.Post("/api/generate", body, "application/json");
        if (!res || res->status != 200)
            return "ERROR: Ollama unavailable. Run: ollama serve";
        return parseResponse(res->body);
    }
};

// =====================================================================
//  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
// =====================================================================

struct DocItem {
    int         id;
    std::string title;
    std::string text;
    std::vector<float> emb;
};

class DocumentDB {
    std::unordered_map<int, DocItem> store;
    HNSW       hnsw;
    BruteForce bf;       // brute force fallback for small sets
    std::mutex mu;
    int nextId = 1;
    int dims   = 0;      // determined from first inserted embedding

public:
    DocumentDB() : hnsw(16, 200) {}

    // Insert one chunk with its pre-computed embedding
    int insert(const std::string& title, const std::string& text,
               const std::vector<float>& emb)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (dims == 0) dims = (int)emb.size();
        DocItem item{nextId++, title, text, emb};
        store[item.id] = item;
        VectorItem vi{item.id, title, "doc", emb};
        hnsw.insert(vi, cosine);
        bf.insert(vi);
        return item.id;
    }

    // Semantic search — returns top-k most similar chunks
    std::vector<std::pair<float, DocItem>> search(
        const std::vector<float>& q, int k, float max_dist = 0.7f)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (store.empty()) return {};
        auto raw = (store.size() < 10)
                   ? bf.knn(q, k, cosine)
                   : hnsw.knn(q, k, 50, cosine);
        std::vector<std::pair<float, DocItem>> out;
        for (auto& [d, id] : raw)
            if (store.count(id) && d <= max_dist) out.push_back({d, store[id]});
        return out;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); hnsw.remove(id); bf.remove(id);
        return true;
    }

    std::vector<DocItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<DocItem> r;
        for (auto& [id, v] : store) r.push_back(v);
        return r;
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }

    int getDims() { return dims; }
};

// =====================================================================
//  DEMO DATA  (60D categorical vectors across 15 categories)
// =====================================================================

// Helper to construct structured 60D categorical vectors
// 15 categories x 4 dimensions each = 60 dimensions
// Dominant category gets strong activations (0.75-0.92)
// Cross-disciplinary ties get mild activations (~0.20)
// Baseline ambient dimensions receive small realistic variations (0.04-0.08)
static std::vector<float> makeVec60(int catIdx, float d0, float d1, float d2, float d3,
                                    int relatedCat = -1, float relWeight = 0.20f) {
    std::vector<float> v(DIMS);
    for (int i = 0; i < DIMS; i++) {
        v[i] = 0.05f + 0.03f * std::sin((float)(catIdx * 5 + i * 2));
    }
    if (relatedCat >= 0 && relatedCat < 15) {
        int rbase = relatedCat * 4;
        v[rbase + 0] = relWeight * 0.95f;
        v[rbase + 1] = relWeight * 0.90f;
        v[rbase + 2] = relWeight * 0.85f;
        v[rbase + 3] = relWeight * 0.80f;
    }
    int base = catIdx * 4;
    v[base + 0] = d0;
    v[base + 1] = d1;
    v[base + 2] = d2;
    v[base + 3] = d3;
    return v;
}

void loadDemo(VectorDB& db) {
    auto dist = getDistFn("cosine");

    // Category 0: Computer Science & Algorithms (Dims 0-3)
    db.insert("Linked List: linear data structure where elements point to subsequent nodes via dynamic pointers", "cs",
        makeVec60(0, 0.90f, 0.85f, 0.72f, 0.68f, 1, 0.15f), dist);
    db.insert("Binary Search Tree: hierarchical tree structure providing O(log n) average search, insertion, and deletion", "cs",
        makeVec60(0, 0.88f, 0.82f, 0.78f, 0.74f, 1, 0.18f), dist);
    db.insert("Dynamic Programming: algorithm design technique solving complex problems via memoized overlapping subproblems", "cs",
        makeVec60(0, 0.82f, 0.76f, 0.88f, 0.80f, 1, 0.22f), dist);
    db.insert("Graph BFS and DFS: fundamental graph traversal algorithms exploring neighbors breadth-first or depth-first", "cs",
        makeVec60(0, 0.85f, 0.80f, 0.75f, 0.82f, 1, 0.14f), dist);
    db.insert("Hash Table: associative data structure mapping keys to buckets using hash functions for O(1) average lookup", "cs",
        makeVec60(0, 0.87f, 0.78f, 0.70f, 0.76f, 1, 0.12f), dist);

    // Category 1: Mathematics & Logic (Dims 4-7)
    db.insert("Calculus: mathematical study of continuous change through derivatives, integrals, and limits", "math",
        makeVec60(1, 0.91f, 0.86f, 0.78f, 0.72f, 2, 0.24f), dist);
    db.insert("Linear Algebra: branch studying vector spaces, linear mappings, matrix transformations, and eigenvectors", "math",
        makeVec60(1, 0.88f, 0.90f, 0.82f, 0.76f, 0, 0.20f), dist);
    db.insert("Probability Theory: mathematical framework analyzing random variables, likelihood distributions, and Bayes theorem", "math",
        makeVec60(1, 0.84f, 0.80f, 0.88f, 0.82f, 9, 0.18f), dist);
    db.insert("Number Theory: pure mathematics examining integer properties, prime factorization, and modular arithmetic", "math",
        makeVec60(1, 0.80f, 0.85f, 0.76f, 0.90f, 0, 0.22f), dist);
    db.insert("Combinatorics: branch of mathematics concerning discrete counting, permutations, and graph enumeration", "math",
        makeVec60(1, 0.86f, 0.78f, 0.84f, 0.80f, 0, 0.25f), dist);

    // Category 2: Physics & Astronomy (Dims 8-11)
    db.insert("Quantum Superposition: quantum principle that physical systems simultaneously occupy multiple states until measured", "physics",
        makeVec60(2, 0.92f, 0.86f, 0.80f, 0.75f, 1, 0.28f), dist);
    db.insert("General Relativity: Einstein's geometric formulation describing gravity as the curvature of spacetime by mass", "physics",
        makeVec60(2, 0.88f, 0.91f, 0.77f, 0.83f, 1, 0.26f), dist);
    db.insert("Thermodynamics Second Law: physical law stating the total entropy of an isolated system increases over time", "physics",
        makeVec60(2, 0.85f, 0.82f, 0.89f, 0.79f, 8, 0.22f), dist);
    db.insert("Standard Model: fundamental quantum field theory classifying quarks, leptons, gauge bosons, and the Higgs field", "physics",
        makeVec60(2, 0.82f, 0.87f, 0.84f, 0.90f, 8, 0.20f), dist);
    db.insert("Electromagnetic Induction: Faraday's law demonstrating that time-varying magnetic flux induces an electric current", "physics",
        makeVec60(2, 0.86f, 0.80f, 0.85f, 0.78f, 1, 0.18f), dist);

    // Category 3: Geology & Earth Sciences (Dims 12-15)
    db.insert("Plate Tectonics: scientific theory that Earth's lithosphere comprises rigid plates driving seismic and volcanic activity", "geology",
        makeVec60(3, 0.90f, 0.84f, 0.81f, 0.75f, 2, 0.16f), dist);
    db.insert("Metamorphic Rock: transformation of existing rock protoliths under extreme thermal heat and tectonic pressure", "geology",
        makeVec60(3, 0.85f, 0.89f, 0.78f, 0.82f, 8, 0.19f), dist);
    db.insert("Sedimentary Stratigraphy: geological discipline analyzing rock layer sedimentation, deposition epochs, and fossil records", "geology",
        makeVec60(3, 0.82f, 0.86f, 0.88f, 0.77f, 7, 0.21f), dist);
    db.insert("Mantle Convection: slow buoyant thermal circulation in Earth's mantle driving continental drift and volcanism", "geology",
        makeVec60(3, 0.88f, 0.80f, 0.84f, 0.89f, 2, 0.20f), dist);
    db.insert("Mohs Hardness Scale: empirical qualitative scale testing scratch resistance of minerals from talc to diamond", "geology",
        makeVec60(3, 0.84f, 0.83f, 0.79f, 0.87f, 8, 0.22f), dist);

    // Category 4: Geopolitics & Diplomacy (Dims 16-19)
    db.insert("Heartland Theory: Mackinder's geopolitical thesis that controlling inner Eurasia commands the global heartland", "geopolitics",
        makeVec60(4, 0.91f, 0.85f, 0.78f, 0.72f, 5, 0.25f), dist);
    db.insert("Bilateral Diplomacy: formal political, economic, and strategic negotiations between two sovereign nation-states", "geopolitics",
        makeVec60(4, 0.86f, 0.89f, 0.82f, 0.76f, 5, 0.22f), dist);
    db.insert("Deterrence Doctrine: military strategy dissuading adversarial aggression through guaranteed devastating retaliation", "geopolitics",
        makeVec60(4, 0.83f, 0.81f, 0.89f, 0.84f, 5, 0.18f), dist);
    db.insert("Exclusive Economic Zone: UN maritime zone granting sovereign rights over marine resources up to 200 nautical miles", "geopolitics",
        makeVec60(4, 0.80f, 0.86f, 0.84f, 0.88f, 9, 0.24f), dist);
    db.insert("Balance of Power: international equilibrium principle preventing any single superpower from achieving absolute hegemony", "geopolitics",
        makeVec60(4, 0.88f, 0.83f, 0.80f, 0.86f, 5, 0.27f), dist);

    // Category 5: World History & Antiquity (Dims 20-23)
    db.insert("Industrial Revolution: historical era of rapid technological shift from manual craft to mechanized steam production", "history",
        makeVec60(5, 0.90f, 0.86f, 0.79f, 0.74f, 9, 0.25f), dist);
    db.insert("Silk Road Trade: ancient continental trade route network connecting Han China, Central Asia, and Rome", "history",
        makeVec60(5, 0.85f, 0.91f, 0.80f, 0.78f, 4, 0.23f), dist);
    db.insert("Peace of Westphalia: 1648 diplomatic treaties establishing modern international sovereign statehood principles", "history",
        makeVec60(5, 0.82f, 0.84f, 0.89f, 0.83f, 4, 0.30f), dist);
    db.insert("Renaissance Humanism: cultural and intellectual movement reviving Greco-Roman antiquity, inquiry, and humanism", "history",
        makeVec60(5, 0.88f, 0.79f, 0.85f, 0.87f, 10, 0.24f), dist);
    db.insert("Magna Carta: 1215 royal charter establishing constitutional limits on monarchical authority and the rule of law", "history",
        makeVec60(5, 0.84f, 0.87f, 0.76f, 0.91f, 4, 0.26f), dist);

    // Category 6: Education & Pedagogy (Dims 24-27)
    db.insert("Socratic Method: pedagogical dialectic inquiry stimulating critical thought through cooperative questioning", "education",
        makeVec60(6, 0.91f, 0.87f, 0.76f, 0.70f, 10, 0.28f), dist);
    db.insert("Bloom's Taxonomy: multi-tiered educational framework classifying cognitive learning objectives from recall to creation", "education",
        makeVec60(6, 0.87f, 0.92f, 0.81f, 0.75f, 10, 0.20f), dist);
    db.insert("Constructivism: educational philosophy where students actively construct knowledge through experiential learning", "education",
        makeVec60(6, 0.83f, 0.80f, 0.89f, 0.82f, 10, 0.22f), dist);
    db.insert("Spaced Repetition: evidence-based learning technique using timed review intervals to maximize mnemonic retention", "education",
        makeVec60(6, 0.85f, 0.84f, 0.79f, 0.88f, 0, 0.19f), dist);
    db.insert("Flipped Classroom: instructional model where lecture materials are studied independently and class time applies concepts", "education",
        makeVec60(6, 0.89f, 0.81f, 0.84f, 0.85f, 0, 0.15f), dist);

    // Category 7: Biology & Life Sciences (Dims 28-31)
    db.insert("CRISPR-Cas9: molecular gene editing technology adapted from bacterial adaptive immune response mechanisms", "biology",
        makeVec60(7, 0.92f, 0.85f, 0.79f, 0.74f, 8, 0.25f), dist);
    db.insert("Cellular Respiration: biochemical metabolic process converting glucose and oxygen into cellular ATP energy", "biology",
        makeVec60(7, 0.87f, 0.90f, 0.81f, 0.76f, 8, 0.27f), dist);
    db.insert("DNA Double Helix: nucleic acid molecular structure composed of antiparallel strands with complementary base pairing", "biology",
        makeVec60(7, 0.84f, 0.81f, 0.88f, 0.82f, 8, 0.24f), dist);
    db.insert("Photosynthesis: anabolic biological process converting solar photon energy into glucose within chloroplasts", "biology",
        makeVec60(7, 0.89f, 0.83f, 0.77f, 0.86f, 8, 0.26f), dist);
    db.insert("Natural Selection: evolutionary mechanism where advantageous hereditary traits increase reproductive fitness", "biology",
        makeVec60(7, 0.86f, 0.88f, 0.83f, 0.89f, 3, 0.18f), dist);

    // Category 8: Chemistry & Molecular Science (Dims 32-35)
    db.insert("Periodic Law: foundational principle organizing elements by increasing atomic number and periodic valence traits", "chemistry",
        makeVec60(8, 0.90f, 0.87f, 0.80f, 0.72f, 2, 0.25f), dist);
    db.insert("Covalent Bonding: chemical bond formed by the cooperative sharing of valence electron pairs between atomic nuclei", "chemistry",
        makeVec60(8, 0.86f, 0.91f, 0.83f, 0.75f, 2, 0.22f), dist);
    db.insert("Acid-Base Neutralization: aqueous chemical reaction between hydronium and hydroxide ions producing salt and water", "chemistry",
        makeVec60(8, 0.83f, 0.79f, 0.89f, 0.81f, 7, 0.20f), dist);
    db.insert("Gibbs Free Energy: thermodynamic state function quantifying maximum reversible work and reaction spontaneity", "chemistry",
        makeVec60(8, 0.88f, 0.84f, 0.77f, 0.86f, 2, 0.28f), dist);
    db.insert("Electronegativity: atomic chemical property measuring the ability of an atom to attract shared bonding electrons", "chemistry",
        makeVec60(8, 0.85f, 0.88f, 0.82f, 0.90f, 2, 0.21f), dist);

    // Category 9: Economics & Global Finance (Dims 36-39)
    db.insert("Supply and Demand: microeconomic market model determining equilibrium pricing based on consumer and producer choices", "economics",
        makeVec60(9, 0.91f, 0.86f, 0.77f, 0.71f, 1, 0.18f), dist);
    db.insert("Monetary Policy: central bank interventions adjusting interest rates and liquidity to stabilize inflation and employment", "economics",
        makeVec60(9, 0.87f, 0.90f, 0.82f, 0.76f, 4, 0.24f), dist);
    db.insert("Comparative Advantage: economic theory demonstrating gains from trade when producing at lower opportunity cost", "economics",
        makeVec60(9, 0.84f, 0.81f, 0.88f, 0.83f, 4, 0.22f), dist);
    db.insert("Fiscal Stimulus: macroeconomic intervention using government expenditure or tax policy to counter economic recession", "economics",
        makeVec60(9, 0.88f, 0.83f, 0.79f, 0.87f, 4, 0.26f), dist);
    db.insert("Price Elasticity: economic measure quantifying the responsiveness of product quantity demanded to price shifts", "economics",
        makeVec60(9, 0.85f, 0.87f, 0.84f, 0.89f, 1, 0.21f), dist);

    // Category 10: Philosophy & Ethics (Dims 40-43)
    db.insert("Categorical Imperative: Kantian deontological moral law demanding actions be universalizable as maxims", "philosophy",
        makeVec60(10, 0.90f, 0.85f, 0.78f, 0.72f, 6, 0.23f), dist);
    db.insert("Utilitarianism: normative consequentialist ethical theory seeking the greatest happiness for the greatest number", "philosophy",
        makeVec60(10, 0.86f, 0.91f, 0.81f, 0.75f, 9, 0.20f), dist);
    db.insert("Epistemology: core branch of philosophical study investigating the justification, truth, and limits of knowledge", "philosophy",
        makeVec60(10, 0.83f, 0.80f, 0.89f, 0.82f, 6, 0.26f), dist);
    db.insert("Existentialism: philosophical inquiry exploring authentic individuality, freedom of choice, and existential angst", "philosophy",
        makeVec60(10, 0.87f, 0.82f, 0.84f, 0.88f, 11, 0.25f), dist);
    db.insert("Stoicism: ancient Hellenistic philosophy emphasizing rational virtue, emotional equanimity, and accepting nature", "philosophy",
        makeVec60(10, 0.84f, 0.88f, 0.79f, 0.86f, 11, 0.21f), dist);

    // Category 11: Literature & Linguistics (Dims 44-47)
    db.insert("Stream of Consciousness: modern narrative technique portraying a character's continuous unfiltered thought progression", "literature",
        makeVec60(11, 0.91f, 0.84f, 0.77f, 0.73f, 10, 0.24f), dist);
    db.insert("Dramatic Irony: storytelling device where literary audiences possess critical insights withheld from characters", "literature",
        makeVec60(11, 0.86f, 0.89f, 0.82f, 0.77f, 12, 0.19f), dist);
    db.insert("Iambic Pentameter: poetic metrical cadence featuring five feet of alternating unstressed and stressed syllables", "literature",
        makeVec60(11, 0.82f, 0.81f, 0.88f, 0.83f, 12, 0.22f), dist);
    db.insert("Magical Realism: narrative literary genre combining hyper-realistic settings with fantastical mythical elements", "literature",
        makeVec60(11, 0.88f, 0.83f, 0.79f, 0.87f, 12, 0.27f), dist);
    db.insert("Hero's Journey: comparative mythological narrative archetype detailing the classic departure, trial, and return", "literature",
        makeVec60(11, 0.85f, 0.87f, 0.83f, 0.90f, 5, 0.25f), dist);

    // Category 12: Art & Architecture (Dims 48-51)
    db.insert("Chiaroscuro: visual art technique utilizing stark tonal contrasts of light and shadow to model dramatic volume", "art",
        makeVec60(12, 0.90f, 0.86f, 0.78f, 0.72f, 11, 0.22f), dist);
    db.insert("Bauhaus Movement: modern aesthetic doctrine integrating craftsmanship, industrial functionalism, and fine arts", "art",
        makeVec60(12, 0.85f, 0.91f, 0.80f, 0.76f, 5, 0.20f), dist);
    db.insert("Linear Perspective: mathematical geometric system creating the illusion of spatial three-dimensional depth", "art",
        makeVec60(12, 0.82f, 0.80f, 0.89f, 0.84f, 1, 0.26f), dist);
    db.insert("Impressionism: revolutionary art movement capturing optical qualities of natural light through visible brushwork", "art",
        makeVec60(12, 0.88f, 0.83f, 0.77f, 0.86f, 2, 0.18f), dist);
    db.insert("Gothic Architecture: medieval architectural tradition defined by soaring ribbed vaults, pointed arches, and buttresses", "art",
        makeVec60(12, 0.84f, 0.88f, 0.82f, 0.89f, 5, 0.27f), dist);

    // Category 13: Food & Culinary Arts (Dims 52-55)
    db.insert("Neapolitan Pizza: wood-fired artisan flatbread topped with San Marzano tomatoes, fresh mozzarella, and basil", "food",
        makeVec60(13, 0.90f, 0.86f, 0.78f, 0.72f, 7, 0.15f), dist);
    db.insert("Sushi: traditional Japanese culinary art combining seasoned vinegared rice with pristine raw sashimi", "food",
        makeVec60(13, 0.86f, 0.90f, 0.82f, 0.76f, 7, 0.17f), dist);
    db.insert("Ramen: Japanese noodle broth dish featuring rich tonkotsu or shoyu broth, scallions, nori, and chashu pork", "food",
        makeVec60(13, 0.82f, 0.78f, 0.90f, 0.84f, 7, 0.16f), dist);
    db.insert("Tacos: iconic Mexican street dish composed of warm corn tortillas stuffed with braised meats and salsa", "food",
        makeVec60(13, 0.78f, 0.82f, 0.86f, 0.90f, 7, 0.14f), dist);
    db.insert("Croissant: laminated French viennoiserie pastry folded with European butter to achieve flaky airy layers", "food",
        makeVec60(13, 0.85f, 0.80f, 0.76f, 0.82f, 8, 0.18f), dist);

    // Category 14: Sports & Athletics (Dims 56-59)
    db.insert("Basketball: dynamic team sport centered on dribbling, perimeter shooting, screens, and fast-break points", "sports",
        makeVec60(14, 0.91f, 0.85f, 0.78f, 0.72f, -1), dist);
    db.insert("Football: strategic gridiron sport combining precision forward passing, defensive formations, and touchdowns", "sports",
        makeVec60(14, 0.87f, 0.89f, 0.82f, 0.76f, -1), dist);
    db.insert("Tennis: individual racket sport requiring explosive lateral court agility, topspin volleys, and precise serves", "sports",
        makeVec60(14, 0.83f, 0.80f, 0.88f, 0.82f, -1), dist);
    db.insert("Chess: classical tactical board game involving opening memorization, pawn structures, and mating combinations", "sports",
        makeVec60(14, 0.80f, 0.84f, 0.78f, 0.90f, 0, 0.25f), dist);
    db.insert("Swimming: competitive aquatic sport encompassing synchronized freestyle, backstroke, breaststroke, and butterfly", "sports",
        makeVec60(14, 0.85f, 0.82f, 0.86f, 0.80f, 2, 0.17f), dist);
}

// =====================================================================
//  HTTP SERVER
// =====================================================================

int main() {
    VectorDB   db(DIMS);
    DocumentDB docDB;
    OllamaClient ollama;

    loadDemo(db);

    // Check Ollama at startup (non-fatal)
    bool ollamaUp = ollama.isAvailable();
    std::cout << "=== VectorDB Engine ===" << std::endl;
    std::cout << "http://localhost:8080" << std::endl;
    std::cout << db.size() << " demo vectors | " << DIMS << " dims | HNSW+KD-Tree+BruteForce" << std::endl;
    std::cout << "Ollama: " << (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)") << std::endl;
    if (ollamaUp) std::cout << "  embed model: " << ollama.embedModel
                            << "  gen model: "   << ollama.genModel << std::endl;

    httplib::Server svr;

    // CORS preflight
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        cors(res); res.status = 204;
    });

    // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5;
        try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto algo   = req.get_param_value("algo");   if (algo.empty())   algo   = "hnsw";

        auto out = db.search(q, k, metric, algo);
        std::ostringstream ss;
        ss << "{\"results\":[";
        for (size_t i = 0; i < out.hits.size(); i++) {
            if (i) ss << ',';
            auto& h = out.hits[i];
            ss << "{\"id\":"        << h.id
               << ",\"metadata\":"  << jS(h.meta)
               << ",\"category\":"  << jS(h.cat)
               << ",\"distance\":"  << std::fixed << std::setprecision(6) << h.dist
               << ",\"embedding\":" << jVec(h.emb) << '}';
        }
        ss << "],\"latencyUs\":" << out.us
           << ",\"algo\":"       << jS(out.algo)
           << ",\"metric\":"     << jS(out.metric) << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string meta, cat; std::vector<float> emb;
        if (!parseBody(req.body, meta, cat, emb) || (int)emb.size() != DIMS) {
            res.set_content("{\"error\":\"invalid body\"}", "application/json"); return;
        }
        int id = db.insert(meta, cat, emb, getDistFn("cosine"));
        res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
    });

    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id  = std::stoi(req.matches[1]);
        bool ok = db.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto items = db.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < items.size(); i++) {
            if (i) ss << ',';
            auto& v = items[i];
            ss << "{\"id\":"        << v.id
               << ",\"metadata\":"  << jS(v.metadata)
               << ",\"category\":"  << jS(v.category)
               << ",\"embedding\":" << jVec(v.emb) << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5; try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto b = db.benchmark(q, k, metric);
        std::ostringstream ss;
        ss << "{\"bruteforceUs\":" << b.bfUs << ",\"kdtreeUs\":" << b.kdUs
           << ",\"hnswUs\":"       << b.hnswUs << ",\"itemCount\":" << b.n << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/hnsw-info", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto gi = db.hnswInfo();
        std::ostringstream ss;
        ss << "{\"topLayer\":" << gi.topLayer << ",\"nodeCount\":" << gi.nodeCount
           << ",\"nodesPerLayer\":[";
        for (size_t i = 0; i < gi.nodesPerLayer.size(); i++) {
            if (i) ss << ','; ss << gi.nodesPerLayer[i];
        }
        ss << "],\"edgesPerLayer\":[";
        for (size_t i = 0; i < gi.edgesPerLayer.size(); i++) {
            if (i) ss << ','; ss << gi.edgesPerLayer[i];
        }
        ss << "],\"nodes\":[";
        for (size_t i = 0; i < gi.nodes.size(); i++) {
            if (i) ss << ',';
            auto& n = gi.nodes[i];
            ss << "{\"id\":" << n.id << ",\"metadata\":" << jS(n.metadata)
               << ",\"category\":" << jS(n.category) << ",\"maxLyr\":" << n.maxLyr << '}';
        }
        ss << "],\"edges\":[";
        for (size_t i = 0; i < gi.edges.size(); i++) {
            if (i) ss << ',';
            auto& e = gi.edges[i];
            ss << "{\"src\":" << e.src << ",\"dst\":" << e.dst << ",\"lyr\":" << e.lyr << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

    // POST /doc/insert  {"title":"...","text":"..."}
    // Chunks the text, embeds each chunk via Ollama, stores in DocumentDB
    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto title = extractStr(req.body, "title");
        auto text  = extractStr(req.body, "text");
        if (title.empty() || text.empty()) {
            res.set_content("{\"error\":\"need title and text\"}", "application/json"); return;
        }

        auto chunks = chunkText(text, 250, 30);
        std::vector<int> ids;

        for (int i = 0; i < (int)chunks.size(); i++) {
            auto emb = ollama.embed(chunks[i]);
            if (emb.empty()) {
                res.set_content(
                    "{\"error\":\"Ollama unavailable. "
                    "Install from https://ollama.com then run: "
                    "ollama pull nomic-embed-text && ollama pull llama3.2\"}",
                    "application/json");
                return;
            }
            std::string chunkTitle = (chunks.size() > 1)
                ? title + " [" + std::to_string(i+1) + "/" + std::to_string(chunks.size()) + "]"
                : title;
            ids.push_back(docDB.insert(chunkTitle, chunks[i], emb));
        }

        std::ostringstream ss;
        ss << "{\"ids\":[";
        for (size_t i = 0; i < ids.size(); i++) { if (i) ss << ','; ss << ids[i]; }
        ss << "],\"chunks\":" << chunks.size()
           << ",\"dims\":"    << docDB.getDims() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // DELETE /doc/delete/123
    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id  = std::stoi(req.matches[1]);
        bool ok = docDB.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    // GET /doc/list
    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto docs = docDB.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < docs.size(); i++) {
            if (i) ss << ',';
            // Truncate text preview to 120 chars
            std::string preview = docs[i].text.substr(0, 120);
            if (docs[i].text.size() > 120) preview += "…";
            ss << "{\"id\":" << docs[i].id
               << ",\"title\":" << jS(docs[i].title)
               << ",\"preview\":" << jS(preview)
               << ",\"words\":"  << (int)std::count(docs[i].text.begin(), docs[i].text.end(), ' ') + 1
               << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    // POST /doc/search {"question":"...","k":3}
    // Fast retrieval for the UI visualizer
    svr.Post("/doc/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto question = extractStr(req.body, "question");
        int  k        = extractInt(req.body, "k", 3);
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json"); return;
        }

        auto qEmb = ollama.embed(question);
        if (qEmb.empty()) {
            res.set_content("{\"error\":\"Ollama unavailable\"}", "application/json"); return;
        }

        auto hits = docDB.search(qEmb, k);

        std::ostringstream ss;
        ss << "{\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // POST /doc/ask  {"question":"...","k":3}
    // Full RAG pipeline: embed → retrieve → generate
    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto question = extractStr(req.body, "question");
        int  k        = extractInt(req.body, "k", 3);
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json"); return;
        }

        // Step 1: parse conversation history (sliding window up to 20 messages)
        auto history = parseChatHistory(req.body);

        // Step 2: embed the question
        auto qEmb = ollama.embed(question);
        if (qEmb.empty()) {
            res.set_content("{\"error\":\"Ollama unavailable\"}", "application/json"); return;
        }

        // Step 3: retrieve top-k relevant chunks
        auto hits = docDB.search(qEmb, k);

        // If no direct hits and there is conversation history, try a contextualized query
        // using the most recent user question for retrieval:
        if (hits.empty() && !history.empty()) {
            std::string lastUserQuery = "";
            for (auto it = history.rbegin(); it != history.rend(); ++it) {
                if (it->role == "user") {
                    lastUserQuery = it->content;
                    break;
                }
            }
            if (!lastUserQuery.empty()) {
                auto cqEmb = ollama.embed(lastUserQuery + " " + question);
                if (!cqEmb.empty()) {
                    hits = docDB.search(cqEmb, k);
                }
            }
        }

        std::ostringstream ctx;
        for (int i = 0; i < (int)hits.size(); i++) {
            ctx << "[" << (i+1) << "] " << hits[i].second.title << ":\n"
                << hits[i].second.text << "\n\n";
        }

        std::ostringstream hist;
        if (!history.empty()) {
            hist << "CONVERSATION HISTORY:\n";
            for (const auto& msg : history) {
                if (msg.role == "user") {
                    hist << "User: " << msg.content << "\n";
                } else {
                    hist << "Assistant: " << msg.content << "\n";
                }
            }
            hist << "\n";
        }

        std::string prompt =
            "You are a helpful AI assistant. Answer the user's question directly, accurately, and conversationally.\n\n";

        if (!hist.str().empty()) {
            prompt += hist.str();
        }

        if (!ctx.str().empty()) {
            prompt += "DOCUMENT CONTEXT:\n" + ctx.str();
        }

        prompt +=
            "CURRENT QUESTION: " + question + "\n\n"
            "INSTRUCTIONS:\n"
            "1. Maintain dialogue context: You must remember and refer to the CONVERSATION HISTORY when answering follow-up questions, pronouns (like 'he', 'she', 'it', 'they', 'that'), or queries about prior topics.\n"
            "2. If relevant information is found in DOCUMENT CONTEXT, incorporate it.\n"
            "3. If the user asks about previous messages or asks a question not in DOCUMENT CONTEXT, answer based on the CONVERSATION HISTORY and your general knowledge.\n"
            "4. Answer directly, naturally, and conversationally without meta-commentary like 'According to the context'.\n\n"
            "Assistant:";

        // Step 4: generate answer
        auto answer = ollama.generate(prompt);

        // Step 5: return everything
        std::ostringstream ss;
        ss << "{\"answer\":" << jS(answer)
           << ",\"model\":"  << jS(ollama.genModel)
           << ",\"historyCount\":" << history.size()
           << ",\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"text\":"     << jS(hits[i].second.text)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "],\"docCount\":" << docDB.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // GET /status
    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        bool up = ollama.isAvailable();
        std::ostringstream ss;
        ss << "{\"ollamaAvailable\":"  << (up ? "true" : "false")
           << ",\"embedModel\":"       << jS(ollama.embedModel)
           << ",\"genModel\":"         << jS(ollama.genModel)
           << ",\"docCount\":"         << docDB.size()
           << ",\"docDims\":"          << docDB.getDims()
           << ",\"demoDims\":"         << DIMS
           << ",\"demoCount\":"        << db.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss;
        ss << "{\"count\":"      << db.size()
           << ",\"dims\":"       << DIMS
           << ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
           << ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        res.set_content(ss.str(), "application/json");
    });

    // Serve index.html
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream f("index.html");
        if (!f.is_open()) { res.status = 404; return; }
        res.set_content(
            std::string(std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>()),
            "text/html");
    });

    svr.listen("0.0.0.0", 8080);
    return 0;
}
