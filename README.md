# VectorDB

An in-memory Vector Database and Retrieval-Augmented Generation (RAG) engine built from scratch in modern C++ (C++17) with an interactive web interface.

The project implements **Hierarchical Navigable Small World (HNSW)**, **KD-Tree**, and **Brute Force (Flat)** vector indexing algorithms side-by-side, providing direct benchmarking capabilities alongside a complete local RAG pipeline powered by Ollama.

---

## System Architecture

```
User Query + Dialogue History (up to 20 messages)
         │
         ├──► Contextual Query Expansion (if follow-up question lacks hits)
         │
         ▼
Ollama Embedding Service (nomic-embed-text)  -->  Generates 768-dimensional dense vector
         │
         ▼
HNSW Index (C++)                             -->  Multilayer proximity graph indexing & ANN search
         │
         ▼
Retrieved Context Chunks + Sliding History   -->  Structured prompt assembly with stop tokens
         │
         ▼
Ollama Generation Service (llama3.2)         -->  Synthesizes context-grounded conversational response
         │
         ▼
Browser Client                               -->  Typewriter rendering + localStorage persistence
```

---

## Core Capabilities

| Component | Technical Details |
|---|---|
| **Multi-Disciplinary Search Space** | **60-dimensional dense coordinate system** spanning 15 academic and humanistic categories (75 semantic anchor vectors) with dominant, cross-disciplinary, and ambient activations |
| **Indexing Algorithms** | **HNSW** (O(log N) approximate search), **KD-Tree** (exact axis-aligned partitioning), **Brute Force** (O(N) exact baseline) |
| **Distance Metrics** | Cosine Similarity, Euclidean Distance (L2), Manhattan Distance (L1) |
| **Dimensionality Reduction** | Principal Component Analysis (PCA) projection to 2D for live vector space visualization |
| **Interactive Visualizer** | 2D PCA scatter plot with node-click search, 15-category legend with hover glow, quick domain query chips, and 60D coordinate inspector |
| **Document Processing** | Automatic sliding-window chunking (250 words per chunk with overlap) and dense embedding ingestion via Ollama |
| **Conversational Context Engine** | Sliding-window memory buffer (up to 20 messages / 10 multi-turn exchanges), pronoun resolution, and contextual search fallback |
| **RAG Pipeline** | Context-injected prompt synthesis queried against local LLM instances (`llama3.2`) with anti-hallucination stop tokens |
| **Session Persistence** | Automatic browser `localStorage` synchronization preserving multi-turn chat threads across page reloads |
| **HTTP Interface** | Embedded REST API providing complete CRUD, search, inspection, conversation, and benchmark endpoints |

---

## Prerequisites

Ensure the following dependencies are installed and available in your environment:

1. **C++17 Compiler**: GCC / MinGW-w64 (via MSYS2 on Windows) or Clang
2. **Ollama**: Local model runtime for embeddings and generation
   - Embedding Model: `nomic-embed-text`
   - Generation Model: `llama3.2` (or `llama3.2:1b` for lower-resource environments)

---

## Setup and Installation

### 1. Configure Local Models (Ollama)

Start the Ollama daemon and pull the required models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

Verify that the models are installed:

```powershell
ollama list
```

> **Note**: An environment with at least 8 GB of system RAM is recommended. Both models combined utilize approximately 3 GB of memory.

### 2. Compile the Server

Compile the C++ backend using GCC with C++17 support and level 2 optimizations:

```powershell
g++ -std=c++17 -O2 -static -static-libgcc -static-libstdc++ main.cpp -o db -lws2_32
```

*Note: The `-lws2_32` flag links the Windows Sockets API (`Winsock2`) required for the HTTP server. The `-static` flags bundle standard C++ runtime libraries so the binary runs standalone.*

### 3. Launch the Application

1. Ensure Ollama is running (`ollama serve` or active in the system tray).
2. Start the VectorDB server:

```powershell
./db
```

Upon successful startup, the server logs initialization metrics and starts listening on port 8080:

```text
=== VectorDB Engine ===
http://localhost:8080
75 demo vectors | 60 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

3. Open your web browser and navigate to:
```
http://localhost:8080
```

---

## User Interface Guide

### Vector Search and Benchmarking
- **60-Dimensional Reference Space**: Query semantic concepts across pre-loaded 60-dimensional reference vectors spanning 15 domains (CS, Mathematics, Physics, Geology, Geopolitics, History, Education, Biology, Chemistry, Economics, Philosophy, Literature, Art, Food, and Sports).
- **Quick-Query Domain Chips**: Click one-touch chips below the search input (`Physics`, `Geology`, `History`, `Education`, `Biology`, `Literature`, `Economics`, `Art`) to instantly execute representative queries and observe real-time dimensional activations.
- **Interactive 15-Category Legend**: Hovering over any category row dynamically pulses and highlights all points belonging to that category across the 2D PCA canvas; clicking a category immediately populates a domain query and triggers search.
- **Interactive 2D PCA Scatter Plot**: Visualizes the global distribution of 75 vectors and ingested documents projected from high dimensions down to 2D. Hover over any point to inspect metadata; click any point to populate its description into the search bar and run nearest-neighbor search.
- **60D Query Embedding Inspector**: Real-time canvas bar chart visualizing all 60 coordinates grouped by 15 domain labels (`CS`, `MTH`, `PHY`, `GEO`, `POL`, `HIS`, `EDU`, `BIO`, `CHM`, `ECN`, `PHL`, `LIT`, `ART`, `FOD`, `SPT`). Hover over any individual bar to reveal its exact coordinate value, dimension index, and category mapping.
- **Algorithm Comparison**: Select between **HNSW**, **KD-Tree**, or **Brute Force** algorithms with **Cosine**, **Euclidean**, or **Manhattan** distance metrics.
- **Microsecond Benchmarking**: Execute **Compare All Algos** to run simultaneous lookups across all indexing strategies and view microsecond-level timing comparisons.

### Document Ingestion
- Input arbitrary plain text (articles, lecture notes, or documentation).
- Submissions are tokenized into 250-word chunks with sliding overlap.
- Each chunk is embedded via `nomic-embed-text` into a 768-dimensional vector and indexed into an independent HNSW graph.

### Retrieval-Augmented Generation (RAG)
- Query ingested documents in natural language.
- The system embeds the query, searches the HNSW index for the top-3 matching chunks, and forwards the retrieved context to `llama3.2`.
- View synthesized responses with real-time typewriter rendering, including expandable references displaying the exact context chunks used by the model.
- **Sliding Context Window Memory**: Maintains a conversational memory buffer of up to **20 messages** (multi-turn dialogue history). Older messages slide out automatically once the 20-message threshold is reached, replicating standard production LLM context management with an in-UI status indicator and memory reset control.

---

## 60-Dimensional Multi-Disciplinary Vector Space

The reference search space is engineered with a **60-dimensional dense coordinate system** partitioned across **15 distinct academic and humanistic domains** (4 dedicated dimensions per domain, comprising 75 curated semantic anchor concepts):

| Coordinate Range | Category | Identifier | Key Semantic Dimensions |
|---|---|---|---|
| `Dims 0–3` | Computer Science & Algorithms | `cs` | Data Structures, Complexity Analysis, Search Traversals, Memory Models |
| `Dims 4–7` | Mathematics & Logic | `math` | Infinitesimal Calculus, Linear Algebra & Matrices, Probability, Number Theory |
| `Dims 8–11` | Physics & Astronomy | `physics` | Quantum Mechanics, Spacetime Relativity, Thermodynamics, Electromagnetism |
| `Dims 12–15` | Geology & Earth Sciences | `geology` | Plate Tectonics, Mineralogy, Stratigraphy, Mantle Convection |
| `Dims 16–19` | Geopolitics & Diplomacy | `geopolitics` | International Sovereignty, Heartland Strategy, Deterrence, Maritime Law |
| `Dims 20–23` | World History & Antiquity | `history` | Industrialization, Trade Networks, Constitutional Law, Renaissance Humanism |
| `Dims 24–27` | Education & Pedagogy | `education` | Socratic Dialectic, Cognitive Taxonomies, Constructivism, Spaced Retrieval |
| `Dims 28–31` | Biology & Life Sciences | `biology` | Gene Editing (CRISPR), Cellular Respiration, DNA Helices, Natural Selection |
| `Dims 32–35` | Chemistry & Molecular Science | `chemistry` | Periodic Valence, Covalent Bonding, Acid-Base Neutralization, Gibbs Free Energy |
| `Dims 36–39` | Economics & Global Finance | `economics` | Price Elasticity, Central Banking, Comparative Advantage, Fiscal Interventions |
| `Dims 40–43` | Philosophy & Ethics | `philosophy` | Deontological Maxims, Utilitarianism, Epistemology, Existential Authenticity |
| `Dims 44–47` | Literature & Linguistics | `literature` | Stream of Consciousness, Dramatic Irony, Poetic Cadence, Monomyth Archetypes |
| `Dims 48–51` | Art & Architecture | `art` | Chiaroscuro Modeling, Functionalist Bauhaus, Geometric Perspective, Impressionism |
| `Dims 52–55` | Food & Culinary Arts | `food` | Fermentation & Dough, Raw Gastronomy, Broth Extraction, Laminated Pastries |
| `Dims 56–59` | Sports & Athletics | `sports` | Kinetic Ball-handling, Gridiron Strategy, Racket Mechanics, Tactical Chess |

### Coordinate Generation Mechanics
1. **Dominant Channel Activation**: Vectors in a category feature high activations ($0.75 - 0.92$) across their 4 primary coordinate axes.
2. **Latent Interdisciplinary Linkages**: Concepts sharing interdisciplinary relationships carry moderate secondary activations ($\approx 0.20$) in related domains (e.g., Quantum Physics connecting with Pure Mathematics, or Chess connecting with Computer Science).
3. **Ambient Baseline Field**: Dimensions outside primary and secondary clusters retain subtle sinusoidal background variation ($0.04 - 0.08$), reflecting real-world dense embeddings.
4. **Interactive 60D Bar Visualizer**: Hovering across bars in the UI reveals the exact coordinate value, dimension index, and category mapping.

---

## Conversational Context History Engine

The RAG subsystem includes an end-to-end multi-turn dialogue architecture designed to provide human-like conversation memory across up to 20 messages (10 complete Q&A exchanges).

### 1. Sliding-Window Memory Buffer (20 Messages)
- **Bounded FIFO Queue**: Chat memory is tracked as a structured sequence of `{ role: "user" | "assistant", content: "..." }` objects.
- **Capacity**: Maintained at `MAX_MEMORY = 20`. Once the dialogue exceeds 20 messages, older messages slide out automatically on both the client (`chatMemory.slice(-MAX_MEMORY)`) and server (`history.erase(...)`).
- **Token Efficiency**: Prevents context explosion while preserving ample context for extended multi-step explorations.

### 2. Contextual Query Retrieval Fallback
- **The Pronoun Problem**: In follow-up inquiries like *"What is his favorite food?"* or *"Where was she born?"*, the query vector contains no mentions of the original subject name. Standard embedding search against the document database often returns 0 chunks because the semantic distance exceeds the threshold (`max_dist = 0.70`).
- **Automatic Query Augmentation**: If direct vector search yields 0 hits and conversation history exists, the engine automatically extracts the most recent user prompt and computes a contextual composite embedding:
  ```cpp
  auto cqEmb = ollama.embed(lastUserQuery + " " + question);
  if (!cqEmb.empty()) hits = docDB.search(cqEmb, k);
  ```
  This guarantees that follow-up questions retrieve the correct document chunks from the HNSW graph without requiring the user to repeat previous context.

### 3. Escape-Safe C++ JSON Parser
- The backend features a zero-dependency C++ parser (`parseChatHistory`) that extracts message roles and content directly from the raw HTTP payload.
- **Escape Character Skipping**: By properly stepping past backslash-escaped characters (`\"`, `\\`, `\n`), the parser reliably handles code blocks, markdown quotations, and JSON snippets without misinterpreting quotes as string boundaries.

### 4. Prompt Assembly & Anti-Hallucination Guardrails
The prompt delivered to `llama3.2` is structured hierarchically:
```text
You are a helpful AI assistant. Answer the user's question directly, accurately, and conversationally.

CONVERSATION HISTORY:
User: <previous question>
Assistant: <previous answer>

DOCUMENT CONTEXT:
[1] <Document Title>:
<Retrieved text chunk>

CURRENT QUESTION: <user question>

INSTRUCTIONS:
1. Maintain dialogue context: Refer to CONVERSATION HISTORY for follow-up questions, pronouns, or prior topics.
2. If relevant information is found in DOCUMENT CONTEXT, incorporate it.
3. Answer directly, naturally, and conversationally without meta-commentary like "According to context".

Assistant:
```
- **Stop Sequences**: The backend configures Ollama generation with `"options": {"stop": ["\nUser:", "\nHuman:"]}` to prevent the model from hallucinating subsequent dialogue turns.

### 5. Frontend Streaming & Session Persistence
- **Persistent Message Stream**: The UI renders an ongoing chat thread with distinct user speech bubbles and assistant cards rather than clearing previous Q&A turns.
- **Unique Turn Scoping**: Each exchange generates a unique turn ID (`t_${timestamp}_${rand}`), isolating DOM nodes for typewriter text and chunk-expansion accordions (`#ctx-${turnId}-${i}`) to eliminate cross-message collisions.
- **LocalStorage Sync**: Dialogue turns and rendered message trees are persisted to browser `localStorage` (`vector_rag_memory` and `vector_rag_html`), surviving browser refreshes and tab re-openings.
- **Visual Memory Badge**: The header features a live status pill (`Memory: X / 20`), turning green when context is active and alerting in orange when the sliding window is engaged.
- **One-Click Reset**: The `Reset Memory` button clears both client state, `localStorage`, and the server session window on demand.
- **Keyboard Shortcut**: Pressing <kbd>Enter</kbd> submits the query; <kbd>Shift</kbd>+<kbd>Enter</kbd> inserts a newline.

---

## REST API Reference

The server exposes a RESTful HTTP API on `http://localhost:8080`.

### Index and Search Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | Performs K-Nearest Neighbor (K-NN) search against the 60D reference database |
| `POST` | `/insert` | Inserts a new 60D vector into the reference index (`{"metadata":"...","category":"...","embedding":[60 floats]}`) |
| `DELETE` | `/delete/:id` | Deletes a reference vector by ID |
| `GET` | `/items` | Returns all reference vectors in the 60D database |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Executes latency comparison across HNSW, KD-Tree, and Brute Force using a 60D vector |
| `GET` | `/hnsw-info` | Returns layer distributions and connectivity metrics for the HNSW graph |
| `GET` | `/stats` | Returns database element counts (75 items), dimensionality (`dims: 60`), and algorithms |

### Document and RAG Endpoints

| Method | Endpoint | Request Payload | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"string","text":"string"}` | Chunks, embeds, and indexes document text via `nomic-embed-text` |
| `GET` | `/doc/list` | None | Lists all indexed document chunks |
| `DELETE` | `/doc/delete/:id` | None | Removes a specific document chunk |
| `POST` | `/doc/ask` | `{"question":"string","k":3,"history":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}` | Executes full RAG pipeline (search + context + 20-message sliding memory + LLM generation) |
| `GET` | `/status` | None | Reports Ollama connectivity, loaded models, and database counts |

### API Usage Examples

#### Nearest Neighbor Search via cURL (60-Dimensional Query)
```powershell
# Construct a 60D vector (e.g. activating Physics: Dims 8-11)
$vec = (0..59 | ForEach-Object { if ($_ -ge 8 -and $_ -le 11) { 0.90 } else { 0.05 } }) -join ','
curl "http://localhost:8080/search?v=$vec&k=3&metric=cosine&algo=hnsw"
```
Or directly passing the 60 comma-separated floats:
```bash
curl "http://localhost:8080/search?v=0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.92,0.86,0.80,0.75,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05&k=3&metric=cosine&algo=hnsw"
```

#### Multi-Turn RAG Query with Conversation History via cURL
```powershell
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{
    "question": "What is his favorite food?",
    "k": 3,
    "history": [
      {"role": "user", "content": "My brother John loves sushi."},
      {"role": "assistant", "content": "Got it! John loves sushi."}
    ]
  }'
```

#### Response Format
```json
{
  "answer": "John's favorite food is sushi.",
  "model": "llama3.2",
  "historyCount": 2,
  "contexts": [],
  "docCount": 0
}
```

---

## Directory Structure

```text
VectorDB/
├── main.cpp        # Core engine: 60D multi-disciplinary reference space, 768D document RAG, HNSW/KD-Tree/BruteForce, REST API
├── httplib.h       # Single-header C++ HTTP/HTTPS server library (yhirose/cpp-httplib)
├── index.html      # Modern UI: 2D PCA scatter plot, 15-category legend, 60D coordinate inspector, multi-turn chat
├── db.exe          # Standalone compiled executable binary (C++17, MinGW-w64)
└── README.md       # Technical architecture, multi-disciplinary taxonomy, and API documentation
```

---

## Algorithmic Overview

### Hierarchical Navigable Small World (HNSW)
HNSW organizes high-dimensional vectors into a hierarchical graph structure inspired by skip lists. 
- **Structure**: Bottom layer (Layer 0) contains all elements with dense connectivity. Each ascending layer contains an exponentially decaying subset of elements with longer-range links.
- **Search**: Greedy routing starts at the sparse top layer to traverse large regions of vector space rapidly, progressively refining search candidates upon descending to lower layers.
- **Characteristics**: Achieves logarithmic O(log N) search complexity and maintains high recall across both moderate ($d = 60$) and high-dimensional spaces (e.g., 768D).

### K-Dimensional Tree (KD-Tree)
A binary space partitioning tree designed for exact Euclidean vector searches.
- **Structure**: Recursively partitions space along median points, cycling through dimensions at each tree depth.
- **Characteristics**: Highly effective for low-to-moderate dimensional spaces ($d \le 60$). In high-dimensional spaces (such as dense LLM embeddings at 768D), branch pruning diminishes due to the curse of dimensionality, causing query times to approach O(N) and illustrating why HNSW approximate graph indexing is preferred for production dense retrieval.

---

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `Ollama: OFFLINE` status | Ollama service is not running | Start the service using `ollama serve` or launch the desktop application. |
| Slow initial embedding response | Model weights are being loaded into memory | Allow 1-2 minutes for the initial model cold-start. |
| `g++: command not found` | Compiler binary directory is not in system PATH | Add the compiler directory (e.g., `C:\msys64\ucrt64\bin`) to the system `Path` variable. |
| `Port 8080 already in use` | Another process is bound to port 8080 | Identify and terminate the conflicting process using `netstat -ano \| findstr 8080` and `taskkill /PID <PID> /F`. |
| High LLM inference latency | Large parameter count for host CPU | Switch to a smaller model variant such as `llama3.2:1b`. |
| Conversation forgotten after 20 messages | Normal sliding window behavior | Older turns beyond 20 messages slide out to keep prompt size bounded. Click **Reset Memory** to start a clean thread or adjust `MAX_MEMORY`. |
| Chat history cleared or lost on refresh | LocalStorage disabled in browser | Ensure browser allows local storage for `localhost:8080`. |

### Configuring an Alternative LLM Model
To use a lighter model variant for faster inference:

1. Pull the target model:
   ```powershell
   ollama pull llama3.2:1b
   ```
2. Update the model identifier in [main.cpp](file:///e:/Vector_Databse_Project/main.cpp):
   ```cpp
   std::string genModel = "llama3.2:1b";
   ```
3. Recompile and restart the server.
