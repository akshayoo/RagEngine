import torch
import faiss
import pickle
import gc
import os
import timeit
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer


mistral7b = "./test_models/models--unsloth--mistral-7b-instruct-v0.1-bnb-4bit/snapshots/ce41ab8056bfc8c399ae1429276920914c3c295e"
embedding_model = "./test_models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

doc_index = "./systems/doc_index.faiss"
doc_split = "./systems/doc_split_text.pkl"

all_models = {
    "mistral": mistral7b,
}

emb_model = SentenceTransformer(embedding_model, local_files_only=True)
loaded_models = {}

def load_model(model_key):
    """Loads a model only once and caches it."""
    if model_key in loaded_models:
        return loaded_models[model_key]

    model_path = all_models[model_key]
    print(f"Loading {model_key}...")
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', local_files_only=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map='auto')
    loaded_models[model_key] = pipe
    return pipe


def get_context(query):
    """Retrieves context from FAISS index."""
    if not os.path.exists(doc_index) or not os.path.exists(doc_split):
        return "No document index found."
    
    emb_query = emb_model.encode([query]).astype("float32")
    index = faiss.read_index(doc_index)
    with open(doc_split, 'rb') as f:
        split_text = pickle.load(f)
    
    D, I = index.search(emb_query, 3)
    retrieved = [split_text[i] for i in I[0]]
    return "\n\n".join(retrieved)


def generate_answer(model_key, query):
    """Runs the RAG process."""
    try:
        pipe = load_model(model_key)
        context = get_context(query)
        prompt = f"""
        You are a helpful assistant.
        Use the following context to answer the question accurately.
        
        Context:
        {context}

        Question: {query}

        Answer:
        """
        output = pipe(prompt, max_new_tokens=150, temperature=0.2, top_p=0.9)[0]['generated_text']
        if "Answer:" in output:
            output = output.split("Answer:")[-1].strip()
        if not output:
            output = "I'm sorry, I couldn't generate an answer right now."
        return output
    except Exception as e:
        return f"Error: {e}"
    finally:
        torch.cuda.empty_cache()
        gc.collect()
