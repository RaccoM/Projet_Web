# Semantic Web & AI: Knowledge Graph Construction, Reasoning, and RAG

## 📌 Project Overview
This repository contains a complete end-to-end **Knowledge Engineering pipeline**. It covers:

* **Data acquisition** from raw web sources.
* **Named Entity Recognition (NER)**.
* **Knowledge Base alignment** with Wikidata.
* **Ontology creation**.
* **SPARQL-based** graph expansion.
* **Explicit rule-based reasoning** (SWRL).
* **Implicit continuous reasoning** (Knowledge Graph Embeddings via PyKEEN).
* **Semantic Retrieval-Augmented Generation (RAG)** interface powered by a local Small Language Model (SLM).

---

## 💻 Hardware Requirements

* **OS:** Windows / Linux / macOS
* **Hardware:** * Minimum **8GB RAM**.
* **Software:**
    * **Python 3.9+**
    * **Ollama** (for running the local SLM)



## ⚙️ Installation & Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/RaccoM/Projet_Web.git
cd Projet_Web
```

### 2. Set up the Python Environment
I used Google Colab with its GPU to execute the cells of the notebook for faster training.

Make sure you have **Python** installed

**Install Requirements :**
```bash
pip install -r requirements.txt
```

### 3. Install & Download the Local LLM
Make sure you have **Ollama** installed, then dawnload **Llama 3.2** (1B parameters) :

```bash
ollama pull llama3.2:1b
```
![Installing Llama](docs/install_llama.png)


## 🚀 How to Run the Modules

### Phase 1 to 4: Graph Construction, SWRL, and KGE
All data acquisition, alignment, SWRL rule execution, and embedding evaluations are contained within a single Jupyter Notebook.

1. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
2. **Execute the pipeline:** Open `src/Web.ipynb` and run the cells sequentially.

> [!NOTE]
> The data outputs (such as KGE datasets) are already provided in the `data/` folder, except `expanded_kb.nt` which are in the
`src/` folder with the notebook.
---

### Phase 5: Semantic RAG Demo (NL to SPARQL)
The RAG system translates Natural Language into SPARQL queries, executes them against our custom RDF graph, and features an automated self-repair loop for syntax errors.

1. **Prerequisite:** Ensure the **Ollama** service is running in the background.
2. **Run the interactive CLI script:**
   ```bash
   python src/lab_rag_sparql_gen.py
   ```
3. **Usage:** Type your questions directly into the terminal (e.g., *"List all the organizations present in the knowledge graph"*). Type `exit` to quit.

---

## 📸 Demo Screenshot

![Demo Screenshot](docs/screenshot.png)