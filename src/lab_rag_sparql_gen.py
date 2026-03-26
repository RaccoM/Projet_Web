import json
import re
import requests
import rdflib
from pathlib import Path
from rdflib.plugins.sparql.processor import SPARQLResult

# CONFIGURATION
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"

# We get the KB file
SCRIPT_DIR = Path(__file__).parent
KB_FILE = SCRIPT_DIR.parent / "data" / "kg_artifacts" / "expanded_kb.nt"


# CORE FUNCTIONS
def load_graph(file_path: str) -> rdflib.Graph:
    print(f" > Loading Knowledge Graph from {file_path}...")
    g = rdflib.Graph()
    g.parse(file_path, format="nt")
    print(f" > Graph loaded with {len(g)} triples.")
    return g

def extract_schema(g: rdflib.Graph) -> str:
    print(" > Extracting schema (Predicates and Classes)...")
    
    # Extract the most frequent predicates (limited to 50 for LLM context)
    res_p = g.query("""
        SELECT ?p (COUNT(?s) AS ?count) 
        WHERE { ?s ?p ?o } 
        GROUP BY ?p 
        ORDER BY DESC(?count) 
        LIMIT 50
    """)
    predicates = [str(r[0]) for r in res_p]

    # Extract classes (types)
    res_c = g.query("""
        SELECT DISTINCT ?c 
        WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?c } 
        LIMIT 50
    """)
    classes = [str(r[0]) for r in res_c]

    schema_text = "### PREDICATES AVAILABLE ###\n" + "\n".join(predicates) + "\n\n"
    schema_text += "### CLASSES AVAILABLE ###\n" + "\n".join(classes) + "\n"
    
    return schema_text

def ask_local_llm(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Communication error with Ollama: {str(e)}"

# RAG & SPARQL PIPELINE
def extract_sparql_from_text(text: str) -> str:
    match = re.search(r'```sparql(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match_fallback = re.search(r'SELECT.*?(?=LIMIT|$)', text, re.DOTALL | re.IGNORECASE)
    if match_fallback:
        return match_fallback.group(0).strip()
    return text.strip()

def generate_sparql(question: str, schema: str) -> str:
    prompt = f"""You are an expert SPARQL query generator.
Given the following RDF schema (Full URIs for predicates and classes), translate the user's natural language question into a valid SPARQL SELECT query.
ALWAYS use the exact full URIs provided in the schema. Do NOT invent prefixes if they are not defined.
Wrap your query inside ```sparql and ``` tags.

SCHEMA:
{schema}

QUESTION: {question}

SPARQL QUERY:
"""
    raw_answer = ask_local_llm(prompt)
    return extract_sparql_from_text(raw_answer)

def run_sparql(g: rdflib.Graph, query: str):
    try:
        qres: SPARQLResult = g.query(query)
        vars_ = [str(v) for v in qres.vars] if qres.vars else []
        rows = []
        for row in qres:
            rows.append([str(val) for val in row])
        return vars_, rows
    except Exception as e:
        raise Exception(f"SPARQL Execution Error: {str(e)}")

def rag_pipeline(g: rdflib.Graph, schema: str, question: str) -> dict:
    sparql_query = generate_sparql(question, schema)
    
    try:
        vars_, rows = run_sparql(g, sparql_query)
        return {
            "query": sparql_query, 
            "vars": vars_, 
            "rows": rows, 
            "repaired": False, 
            "error": None
        }
    except Exception as err:
        # Self-repair loop
        repair_prompt = f"""The following SPARQL query failed to execute.
Question: {question}
Broken Query:
{sparql_query}

Error Message:
{str(err)}

Schema Context:
{schema}

Please fix the syntax or logic errors and provide the corrected SPARQL query wrapped in ```sparql tags.
"""
        raw_repaired = ask_local_llm(repair_prompt)
        repaired_query = extract_sparql_from_text(raw_repaired)
        
        try:
            vars_, rows = run_sparql(g, repaired_query)
            return {
                "query": repaired_query, 
                "vars": vars_, 
                "rows": rows, 
                "repaired": True, 
                "error": None
            }
        except Exception as err2:
            return {
                "query": repaired_query, 
                "vars": [], 
                "rows": [], 
                "repaired": True, 
                "error": str(err2)
            }

def answer_no_rag(question: str) -> str:
    prompt = f"Answer the following question directly and concisely to the best of your knowledge:\n\nQuestion: {question}"
    return ask_local_llm(prompt)

# CLI INTERFACE
def pretty_print_result(result: dict):
    if result.get("error"):
        print(f"\n > Execution Error: {result['error']}")
    
    print("\n[SPARQL Query Used]")
    print("-" * 40)
    print(result.get("query", "No query generated."))
    print("-" * 40)
    print(f"[Repaired via LLM?] {result.get('repaired')}")
    
    vars_ = result.get("vars", [])
    rows = result.get("rows", [])
    
    if not rows and not result.get("error"):
        print("\n > Query executed successfully but returned 0 results.")
        return

    print("\n[Results]")
    print(" | ".join(vars_))
    print("-" * 40)
    for r in rows[:20]:
        print(" | ".join(r))
    
    if len(rows) > 20:
        print(f"... (showing 20 out of {len(rows)} results)")

def main():
    print("\n ===== Semantic RAG Chatbot - Initialization =====\n")
    
    try:
        g = load_graph(KB_FILE)
    except Exception as e:
        print(f" > Critical Error: Unable to load the graph '{KB_FILE}'.\nDetails: {e}")
        return

    schema = extract_schema(g)
    
    print("\nThe system is ready. Type 'exit' or 'quit' to exit.")
    print("="*60)

    while True:
        question = input("\n > Ask your question: ").strip()
        if question.lower() in ['exit', 'quit']:
            print("Closing the RAG system. Goodbye!")
            break
        if not question:
            continue
            
        print("\n --- BASELINE (Pure LLM without RAG) ---")
        print(answer_no_rag(question))
        
        print("\n --- RAG PIPELINE (SPARQL Generation) ---")
        rag_res = rag_pipeline(g, schema, question)
        pretty_print_result(rag_res)

if __name__ == "__main__":
    main()