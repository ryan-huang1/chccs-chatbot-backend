import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from tqdm import tqdm
from flask import Flask, request, jsonify
import random
import json

# Set the TOKENIZERS_PARALLELISM environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load random questions
with open('school_board_questions.json', 'r') as f:
    random_questions = json.load(f)

# Initialize SentenceTransformer model with larger dimensions
dimensions = 1024
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)

app = Flask(__name__)

class EmbeddingSearch:
    def __init__(self, embedding_file="embeddings.pkl"):
        self.embedding_file = embedding_file
        self.documents = []
        self.embeddings = []

    def load_embeddings(self):
        if os.path.exists(self.embedding_file):
            print("Loading pre-computed embeddings...")
            with open(self.embedding_file, 'rb') as f:
                self.documents, self.embeddings = pickle.load(f)
            return True
        else:
            print("No pre-computed embeddings found.")
            return False

    def save_embeddings(self):
        print("Saving embeddings...")
        with open(self.embedding_file, 'wb') as f:
            pickle.dump((self.documents, self.embeddings), f)

    def add_documents(self, documents):
        print("Encoding documents...")
        new_embeddings = []
        for doc in tqdm(documents, desc="Embedding documents"):
            new_embeddings.append(model.encode(doc))
        
        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)
        self.embeddings = np.array(self.embeddings)
        
        self.save_embeddings()

    def search(self, query, top_k=5):
        query_with_prompt = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = model.encode([query_with_prompt])[0]
        scores = np.dot(self.embeddings, query_embedding)
        top_results = np.argsort(scores)[::-1][:top_k]
        
        return [self.documents[i] for i in top_results]

# Initialize EmbeddingSearch
searcher = EmbeddingSearch()

def process_pdfs(pdf_dir):
    documents = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        filepath = os.path.join(pdf_dir, filename)
        try:
            elements = partition_pdf(filepath)
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                documents.append(chunk.text)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return documents

def add_documents_to_search(documents):
    searcher.add_documents(documents)

def retrieve_relevant_docs(query, n_results=20):
    return searcher.search(query, top_k=n_results)

def chat_with_gpt(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    conversation_history = data.get('conversation', [])

    if user_input.lower() == 'clear':
        return jsonify({"response": "Conversation cleared.", "conversation": []})

    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_docs(user_input)
    
    if not conversation_history:
        # Initialize with system message
        conversation_history = [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Use the relevant information provided to answer questions accurately. Make sure to reference the source of the information, and don't make up information not given to you."
            }
        ]
    
    # Attach relevant information to the user's query
    context = "Relevant information:\n" + "\n".join(relevant_docs)
    enhanced_user_input = f"{user_input}\n\n{context}"

    conversation_history.append({"role": "user", "content": enhanced_user_input})
    
    assistant_response = chat_with_gpt(conversation_history)
    
    # Add the original user input (without the relevant info) to the conversation history
    conversation_history[-1]["content"] = user_input
    conversation_history.append({"role": "assistant", "content": assistant_response})
    
    return jsonify({
        "response": assistant_response,
        "conversation": conversation_history
    })

@app.route('/random_question', methods=['GET'])
def get_random_question():
    random_question = random.choice(random_questions)
    return jsonify({"question": random_question})

if __name__ == "__main__":
    # Load existing embeddings or process PDFs
    if not searcher.load_embeddings():
        print("No existing embeddings found. Processing PDFs...")
        pdf_dir = 'pdf'  # Replace with your PDF directory path
        documents = process_pdfs(pdf_dir)
        add_documents_to_search(documents)
        print(f"Processed and added {len(documents)} document chunks.")
    else:
        print(f"Loaded {len(searcher.documents)} document chunks from existing embeddings.")

    app.run(debug=True)