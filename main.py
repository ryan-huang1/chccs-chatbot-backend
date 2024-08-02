import os
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from tqdm import tqdm

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Chroma client
chroma_client = chromadb.Client()

# Use Chroma's default embedding function (Sentence Transformers)
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Create or get the collection
collection = chroma_client.get_or_create_collection(
    name="rag_documents",
    embedding_function=embedding_function
)

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

def add_documents_to_chroma(documents):
    for i, doc in tqdm(enumerate(documents), desc="Adding to Chroma", total=len(documents)):
        collection.add(
            documents=[doc],
            ids=[f"doc_{i}"]
        )

def retrieve_relevant_docs(query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0]

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
    print("Processing PDFs...")
    
    pdf_dir = 'pdf'  # Replace with your PDF directory path
    documents = process_pdfs(pdf_dir)
    add_documents_to_chroma(documents)
    
    print(f"Processed and added {len(documents)} document chunks to Chroma.")
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