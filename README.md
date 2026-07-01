# Modular Agentic RAG Framework with LLM-as-a-Judge Guardrails

A high-performance, modular Retrieval-Augmented Generation (RAG) system running entirely on local architecture with cloud-backed reasoning optimization. This architecture goes beyond basic sequential pipelines by introducing a multi-tiered verification flow that insulates generation windows from semantic noise and contextual drift.

Built and optimized on an Apple Silicon **Mac M4 Pro** environment.

---

## 🏗️ System Architecture & Workflow

Traditional RAG setups pass raw vector database chunks directly into the LLM context window. This creates a massive point of failure for production scale: **accidental keyword matches introduce semantic noise, leading to model hallucinations and token window bloat.**

This system resolves that bottleneck by executing a **Multi-Tiered Cascading Guardrail Pipeline**:

```text
       [ User Query ]
             │
             ▼
  ┌─────────────────────┐
  │  Stage 1: Recall    │ ──► Local Chroma DB Vector Search (Coarse candidate retrieval)
  └─────────────────────┘
             │
             ▼  [Raw Document Chunks]
  ┌─────────────────────┐
  │  Stage 2: Precision │ ──► Deterministic LLM-as-a-Judge Layer (Semantic quality audit)
  └─────────────────────┘
             │
             ├─► Score >= 3 ──► ✅ Compiled into Strict Context Payload
             └─► Score < 3  ──► ❌ Dropped Programmatically (Noise Mitigation)
             │
             ▼  [Sanitized Context Bundle]
  ┌─────────────────────┐
  │  Stage 3: Synthesis │ ──► High-Throughput MoE Streaming Generation (Groq Core)
  └─────────────────────┘
```

1. **Deterministic Build Phase (`create_database.py`):** Parses raw target documents natively, partitioning files into overlapping sections via a `RecursiveCharacterTextSplitter`. Chunks are vectorized using a local `all-MiniLM-L6-v2` cross-encoder framework and indexed securely onto local disk partitions.
2. **Coarse-Grained Vector Recall:** Queries the localized Chroma DB vector indexes to retrieve candidate text documents matching user query geometries.
3. **Fine-Grained Audit Layer (`judge.py`):** Bypasses unstable tool-calling wrappers by feeding candidates into a deterministic, zero-temperature `llama-3.3-70b-versatile` engine. The judge evaluates semantic quality against an explicit 0-to-5 rubric, returning a native JSON schema structure. Items falling below configured threshold parameters are dropped dynamically.
4. **Context Construction & High-Throughput Stream (`main.py`):** Stitches surviving records into unified XML context structures. The finalized context payload is forwarded to an open-weights Mixture-of-Experts (MoE) pipeline on Groq to stream token arrays immediately back to the client interface window at ~500+ tokens/sec.

---

## ⚡ Core Technical Features

* 🛡️ **Native API JSON Enforcements:** Configured underlying hardware inference options via explicit `response_format={"type": "json_object"}` maps. This eliminates structural output parsing anomalies and `tool_use_failed` errors entirely.
* 📦 **Modular Clean-Pipe Separation:** Decoupled data ingestion engines, programmatic evaluation judges, and generation loops into independent structural units, making the pipeline fully ready to import directly into complex multi-agent state architectures like **LangGraph**.
* 🎨 **Visual CLI Timeline Anchors:** Upgraded console output using clear visual indicators to track the automated data lifecycle pipeline in real-time.

---

## 📂 Project Structure

```text
LANGCHAIN-RAG-PROJECT/
├── chroma/                 # Local vectorized SQLite storage database directories
├── data/
│   └── books/              # Ingestion data directory (e.g., alice_in_wonderland.md)
├── .env                    # Strictly isolated API authentication parameters
├── .gitignore              # Dependency and storage environment exclusion maps
├── create_database.py      # Local disk build and embedding ingestion engine
├── judge.py                # Deterministic LLM-as-a-Judge validation module
├── main.py                 # Real-time stateful chat interaction execution hub
├── pyproject.toml          # Comprehensive dependency matrix configurations
└── uv.lock                 # Unified deterministic environment package locks
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
This project utilizes `uv` for ultra-fast, modern Python virtual environment and dependency tracking management.

Clone the workspace and synchronize the environment state:
```bash
# Verify your local environment and sync required dependencies
uv pip install -r requirements.txt
```

### 2. Configure Environment Parameters
Create a `.env` file in your root workspace root folder:
```text
GROQ_API_KEY=your_groq_production_api_key_here
HF_TOKEN=your_optional_huggingface_read_token_here
```

### 3. Build the Local Knowledge Index
Place your target Markdown text documents inside `data/books/` and run the build file:
```bash
python3 create_database.py
```
*Your terminal screen will render a stylized, step-by-step pipeline tracking timeline illustrating text parsing checkpoints.*

### 4. Initialize the Interactive Chat Engine
Boot your application loop interface to test your retrieval architecture:
```bash
python3 main.py
```

---

## ⚖️ Evaluation Rubric (`RetrievalJudge`)

The underlying local judge scores candidates on an explicit quality scale to assure logical integrity before generation mapping:
* **5/5:** Absolute match; context contains the precise direct answer.
* **3-4/5:** Functional match; text block outlines necessary definitions or surrounding background context.
* **0-2/5:** Out-of-bounds anomaly; rejected automatically as background noise.

---

## 🔮 Future Architecture Roadmap (Planned Improvements)

- [ ] **Parallel Batch Auditing:** Refactor the sequential execution loop to use LangChain's native `.batch()` asynchronous protocol. This will process all candidate documents concurrently in a single multi-threaded transaction, targeting up to a **75% reduction in network round-trip latency**.
- [ ] Integrate local cross-encoder model optimization layers (e.g., `FlashRank`) to execute lightning-fast neural reranking before the judgment gates.
- [ ] Migrate the runtime loop array into stateful node-and-edge graphs inside **LangGraph** to deploy a multi-agent hierarchy system matching Kaizan's core multi-agent deliverables.
