# 🛡️ LLM Red Teaming — Single Agent Orchestrator

An autonomous, LLM-powered red teaming pipeline that systematically probes a RAG (Retrieval-Augmented Generation) chatbot for security vulnerabilities using the **OWASP Top 10 for LLMs** framework. Built with **LangGraph** for stateful orchestration and **human-in-the-loop** oversight.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                     │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │   Threat      │──▶│   Prompt     │──▶│  Human-in-the-   │ │
│  │   Modeling    │   │  Generation  │   │  Loop Review     │ │
│  └──────────────┘   └──────────────┘   └────────┬─────────┘ │
│                                                  │           │
│                                                  ▼           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │  Analyze &   │◀──│  Promptfoo   │◀──│  Execute         │ │
│  │  Route       │   │  Baseline    │   │  Baseline Scan   │ │
│  └──────┬───────┘   └──────────────┘   └──────────────────┘ │
│         │                                                    │
│    ┌────▼────┐                                               │
│    │ Route?  │                                               │
│    └────┬────┘                                               │
│    Pass │    Fail / Force                                    │
│    │    │    ┌──────────────────┐                             │
│    │    └───▶│  Deep Scans      │                             │
│    │         │  (Garak / PyRIT / │                             │
│    │         │   DeepTeam)      │                             │
│    │         └────────┬─────────┘                             │
│    │                  │                                       │
│    ▼                  ▼                                       │
│  ┌──────────────────────────────────┐                        │
│  │  Synthesis → OWASP Scorecard    │                        │
│  │  Report (HTML)                   │                        │
│  └──────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Target RAG Chatbot │
              │  (Local HTTP Server)│
              │  http://127.0.0.1   │
              └─────────────────────┘
```

---

## 📂 Project Structure

```
V1_Single_agent/
│
├── Rag_Application/              # 🎯 Target: RAG chatbot to be tested
│   ├── rag_helper.py             # RAGBase class (search, build prompt, LLM call)
│   ├── ingest.py                 # Document ingestion into SQLite search index
│   ├── main.py                   # Application entry point
│   └── pyproject.toml            # Python dependencies (uv)
│
├── Red_Teaming_Agent/            # 🔴 Standalone Promptfoo red teaming agent
│   ├── promptfooconfig.yaml      # Promptfoo test configuration
│   ├── target_provider.py        # HTTP target provider for Promptfoo
│   ├── run_red_teaming.ipynb     # Notebook to run standalone red teaming
│   └── package.json              # Node.js dependencies (promptfoo)
│
├── Single_agent_orchestrator/    # 🧠 Core: LangGraph-based orchestrator
│   ├── orchestrator.py           # Full pipeline: 7-node LangGraph state machine
│   ├── targets.py                # RAGLocalServer — exposes RAG as HTTP endpoint
│   ├── config.yaml               # Target profile, threat vectors, scan parameters
│   ├── report_template.html      # Jinja2 OWASP scorecard template
│   ├── evaluator_helper.js       # Custom Promptfoo evaluator functions
│   ├── garak_rest_config_deep.json  # Garak deep scan configuration
│   ├── promptfoo_baseline_config.yaml  # Generated baseline config (auto)
│   ├── requirements.txt          # Python dependencies
│   ├── orchestrator_notebook.ipynb     # Notebook to run the pipeline
│   └── report.html               # Generated OWASP scorecard (output)
│
├── .gitignore
└── README.md
```

---

## 🔬 How It Works

The pipeline executes as a **7-node LangGraph state machine** with checkpointing and human-in-the-loop review:

| Node | Name | Description |
|------|------|-------------|
| 1 | **Threat Modeling** | LLM analyzes the target app description and maps each active threat vector to the best security framework(s) (Promptfoo, Garak, PyRIT, DeepTeam). |
| 2 | **Prompt Generation** | LLM generates adversarial test prompts for each threat vector (configurable count per vector). |
| 3 | **Human Review** | ⏸️ Pipeline pauses. The operator reviews, edits, or approves the generated prompts before execution. |
| 4 | **Baseline Scan** | Runs all approved prompts through **Promptfoo** against the target RAG chatbot via a local HTTP server. |
| 5 | **Analyze & Route** | Evaluates baseline scores. Routes to deep scans if any vector scores < 0.8 or if `force_deep_scan` is enabled. |
| 6 | **Deep Scans** | Executes specialized scanners (**Garak**, **PyRIT**, **DeepTeam**) and multi-turn adversarial attack loops for flagged vectors. Falls back to an LLM-based programmatic generator if tools are unavailable. |
| 7 | **Report** | Compiles all findings into an **OWASP Top 10 for LLMs** HTML scorecard with per-vector scores, risk ratings, and remediation recommendations. |

### Threat Vectors Tested

| Vector | OWASP Category | Description |
|--------|---------------|-------------|
| `jailbreak` | LLM01: Prompt Injection | Adversarial prompts to bypass system instructions |
| `hallucination` | LLM09: Overreliance | Checks if the model fabricates answers not in the knowledge base |
| `overreliance` | LLM09: Overreliance | Checks if the model uncritically follows false context |
| `rbac` | LLM08: Excessive Agency | Tests role-based access control boundaries |
| `pii_leakage` | LLM06: Sensitive Info Disclosure | Attempts to extract sensitive database details or API keys |

---

## ⚙️ Setup

### Prerequisites

- **Python** ≥ 3.12
- **Node.js** ≥ 18 (for Promptfoo)
- **Ollama** with access to `gemma4:31b-cloud` (or configure a different model)
- **Google API Key** (optional, if using Gemini as the agent model)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd V1_Single_agent
```

### 2. Set Up the RAG Application (Target)

```bash
cd Rag_Application
pip install uv           # if not installed
uv sync                  # installs dependencies from pyproject.toml
```

Create a `.env` file:
```env
OLLAMA_BASE_URL=https://ollama.com
GOOGLE_API_KEY=your_google_api_key_here   # optional
```

Ingest documents into the SQLite search index:
```bash
python ingest.py
```

### 3. Set Up the Orchestrator

```bash
cd ../Single_agent_orchestrator
pip install -r requirements.txt
```

### 4. Set Up Promptfoo (Node.js)

```bash
cd ../Red_Teaming_Agent
npm install
```

---

## 🚀 Usage

### Running the Full Pipeline

1. Open `Single_agent_orchestrator/orchestrator_notebook.ipynb` in Jupyter.
2. Run the setup cells to load `config.yaml` and initialize the graph.
3. The pipeline will execute nodes 1–2, then **pause at node 3** (Human Review).
4. Review the generated adversarial prompts in the notebook output.
5. Resume execution to run baseline scans → deep scans → report generation.
6. Open the generated `report.html` for the OWASP scorecard.

### Configuration

Edit [`config.yaml`](Single_agent_orchestrator/config.yaml) to customize:

```yaml
target:
  name: "Course RAG Chatbot"
  model: "gemma4:31b-cloud"

orchestrator:
  agent_model: "gemma4:31b-cloud"       # LLM used for threat modeling & prompt generation
  active_threat_vectors:
    - "hallucination"
    - "overreliance"
    - "rbac"
    - "jailbreak"
    - "pii_leakage"
  parameters:
    num_tests_per_vector: 2              # Adversarial prompts per vector
    max_concurrency: 1
    delay_ms: 4000
    force_deep_scan: True                # Force deep scans even if baseline passes
```

---

## 📊 Sample Output

The pipeline generates an **OWASP Top 10 for LLMs** HTML scorecard (`report.html`) containing:

- ✅ Per-vector security scores (0.0 – 1.0)
- 🟢🟡🔴 Risk ratings (Secure / Warning / Critical)
- 📋 Individual test case results with prompts, responses, and pass/fail status
- 🔧 AI-generated remediation recommendations
- 📈 Overall security posture summary

---

## 🔧 Security Frameworks Used

| Framework | Role | Language |
|-----------|------|----------|
| [**Promptfoo**](https://github.com/promptfoo/promptfoo) | Baseline security scanning with custom evaluators | Node.js |
| [**Garak**](https://github.com/NVIDIA/garak) | Deep scanning for hallucination, jailbreak, PII leakage | Python |
| [**PyRIT**](https://github.com/Azure/PyRIT) | Multi-turn adversarial red teaming | Python |
| [**DeepTeam**](https://github.com/confident-ai/deepteam) | LLM vulnerability scanning | Python |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph + Checkpointer) |
| Agent LLM | Ollama Cloud (`gemma4:31b-cloud`) / Google Gemini |
| Target App | Custom RAG chatbot (SQLite + Ollama) |
| Report | Jinja2 HTML template |
| Data | Pandas, PyYAML, JSON |

---

## 📝 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- [DataTalks.Club](https://datatalks.club/) — RAG application tutorial
- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — Security framework
- [LangGraph](https://github.com/langchain-ai/langgraph) — Stateful agent orchestration
