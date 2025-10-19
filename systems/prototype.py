import numpy
import torch
import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import gc
import timeit
import os

mistral7binstructv01bnb4bit = "../test_models/models--unsloth--mistral-7b-instruct-v0.1-bnb-4bit/snapshots/ce41ab8056bfc8c399ae1429276920914c3c295e"
phi3mini4kinstruct = "../test_models/models--microsoft--phi-3-mini-4k-instruct/snapshots/0a67737cc96d2554230f90338b163bc6380a2a85"
gemma2bit = "../test_models/models--google--gemma-2b-it/snapshots/96988410cbdaeb8d5093d1ebdc5a8fb563e02bad"
Qwen2_15BInstruct = "../test_models/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"
embedding_model = "../test_models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

doc_index_path = "./doc_index.faiss"
doc_split_path = "./doc_split_text.pkl"

all_models = [
    mistral7binstructv01bnb4bit, phi3mini4kinstruct, gemma2bit, Qwen2_15BInstruct, embedding_model
]

def select_model(all_models):
    
    print('''
          (1) Mistral-7b-instruct-v0.1-bnb-4bit
          (2) Phi-3-mini-4k-instruct
          (3) Gemma2bit
          (4) Qwen-2_15B-Instruct\n''')
    mod_sel = int(input("Select the model of your choice, Input model number: "))
    
    if mod_sel == 1:rag_model = all_models[0]
    elif mod_sel == 2:rag_model = all_models[1]
    elif mod_sel == 3:rag_model = all_models[2]
    elif mod_sel == 4:rag_model = all_models[3]
    else: print("Please select the correct option")
    
    return rag_model



def model_load(ragmodel):
    
    try:
        pipe = None
        torch.cuda.empty_cache()
        model_start = timeit.default_timer()
        
        tokenizer = AutoTokenizer.from_pretrained(ragmodel, local_files_only= True)
        model = AutoModelForCausalLM.from_pretrained(
            ragmodel,
            torch_dtype=torch.float16,
            device_map='auto',
            local_files_only=True)
        
        pipe = pipeline('text-generation', model = model, tokenizer= tokenizer, device_map= 'auto')
        
        model_fin = timeit.default_timer()
        model_load_elapsed = model_fin - model_start
        print(f"\nTime:{model_load_elapsed}\n")
        
    except Exception as e:
        print(f"The model returned an error: {e}")
    
    return pipe



def query_handle(query):
    
    embedding_model_path = all_models[4]
    emb_model = SentenceTransformer(embedding_model_path, local_files_only= True)
    query_embeddings = emb_model.encode([query]).astype('float32')
    
    return query_embeddings



def text_index_handle(doc_index, doc_split, emb_query):
    
    if not os.path.exists(doc_index):
        raise FileNotFoundError(f"FAISS index not found at: {doc_index}")
    if not os.path.exists(doc_split) or os.path.getsize(doc_split) == 0:
        raise FileNotFoundError(f"Pickle file missing or empty: {doc_split}")
    
    index = faiss.read_index(doc_index)
    with open(doc_split, 'rb') as f:
        split_text = pickle.load(f)
        
    k = 3 
    D, I = index.search(emb_query, k)
    retrieved = [split_text[i] for i in I[0]]
    context = "\n\n".join(retrieved)
    
    return context



def output_h(context, pipe, query):
    
    prompt = f"""
    You are a helpful assistant.
    Use the following context to answer the question.
        
    Context:
    {context}

    Question: {query}

    Answer:
    """

    response = pipe(
        prompt,
        max_new_tokens=300,
        temperature=0.3,
        top_p=0.9
    )[0]['generated_text']

    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    return f"\nQuestion: {query}\n\nAnswer: {response}"



def clear_cache():
    try:
        torch.cuda.empty_cache()   
        gc.collect()               
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()  
    except:
        pass



def main():
    
    model_path = select_model(all_models)
    model_pipe = model_load(model_path)
    
    
    while True:
        user_query = input("\nAsk a question(type e to exit, c to change model): ")
        if user_query.lower() == 'e':
            clear_cache()
            print("\nEnding chat Bye......")
            break
            
        elif user_query.lower() == 'c':
            clear_cache()
            print("\nChanging model")
            model_path = select_model(all_models)
            model_pipe = model_load(model_path)
            continue
        
        ans_start = timeit.default_timer()
        
        embedded_query = query_handle(user_query)
        context = text_index_handle(doc_index_path, doc_split_path, embedded_query)
        
        output = output_h(context, pipe= model_pipe, query= user_query)
        
        ans_del = timeit.default_timer()
        ans_el_tm = ans_del - ans_start
        
        print(f"{output}\n Time: {ans_el_tm}")
        clear_cache()
        


if __name__ == "__main__":
    main()
