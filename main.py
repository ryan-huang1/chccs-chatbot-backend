import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from tqdm import tqdm

# Set the TOKENIZERS_PARALLELISM environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize SentenceTransformer model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

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
        query_embedding = model.encode([query])[0]
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

def retrieve_relevant_docs(query, n_results=5):
    return searcher.search(query, top_k=n_results)

def chat_with_gpt(messages, relevant_docs):
    try:
        # Prepend relevant documents to the system message
        context = "Relevant information:\n" + "\n".join(relevant_docs)
        messages[0]["content"] += f"\n\n{context}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    print("Welcome to the RAG-enhanced GPT-4o mini Chatbot!")
    
    # Try to load existing embeddings
    if not searcher.load_embeddings():
        print("No existing embeddings found. Processing PDFs...")
        pdf_dir = 'pdf'  # Replace with your PDF directory path
        documents = process_pdfs(pdf_dir)
        add_documents_to_search(documents)
        print(f"Processed and added {len(documents)} document chunks.")
    else:
        print(f"Loaded {len(searcher.documents)} document chunks from existing embeddings.")

    print("Type 'quit' to exit the chat.")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the relevant information provided to answer questions accurately."}
    ]
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Retrieve relevant documents
        relevant_docs = retrieve_relevant_docs(user_input)
        
        messages.append({"role": "user", "content": user_input})
        
        assistant_response = chat_with_gpt(messages, relevant_docs)
        print("Assistant:", assistant_response)
        print()  # Add a blank line for better readability
        
        messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()