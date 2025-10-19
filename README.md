RagEngine

RagEngine is a Retrieval-Augmented Generation (RAG) based chatbot that answers user queries using information from your own documents.  
It combines FAISS, MiniLM embeddings, and Mistral-7B Instruct to deliver context-aware responses through a Flask-powered web interface.

-----------------------

Features
- Convert documents to embeddings using all-MiniLM-L6-v2
- Store and search document vectors with FAISS
- Retrieve and generate answers using Mistral-7B Instruct (4-bit)
- Simple HTML/CSS web interface
- Real-time Flask backend

------------------------

- Frontend: HTML, CSS, JS  
- Backend: Python (Flask)  
- Embeddings: `all-MiniLM-L6-v2`  
- Vector Store: FAISS  
- Model: `mistral-7b-instruct-v0.1-bnb-4bit`

-------------------------

Installation and running

1. Clone the repository

2. Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate 
   venv\Scripts\activate
   
3. Install dependencies
   pip install -r requirements.txt
   
4. Download required models
    Model files are not included in this repository due to size.
    You can download them manually:
    Embedding model: all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    LLM model: mistral-7b-instruct-v0.1-bnb-4bit (https://huggingface.co/unsloth/mistral-7b-instruct-v0.1-bnb-4bit)
    Or you can download or use a model of your choice. These are the one that I have used since these are light weight and usable for system with less vRAM
    Save or configure their paths appropriately in your code before running.

5. Run the ipynb "systems/rag_conc.ipynb" file replacing with your text to generate the doc_index.faiss and doc_split_text.pkl files based on yuor text. Index and split text file.

6. Run the flask app
     uvicorn main:app --reload

7. Open the chat.html

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Pull requests and suggestions are welcome!
If you find bugs or performance issues, feel free to open an issue.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

License
This repository is licensed under the MIT License.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Contact
Author: Akshay
Email: akshayramesh543@gmail.com

-----------------------   ----------------------- 


