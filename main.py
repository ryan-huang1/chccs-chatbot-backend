import os
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Chroma client
chroma_client = chromadb.Client()

# Use OpenAI's text-embedding-ada-002 model for embeddings
embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# Create or get the collection
collection = chroma_client.get_or_create_collection(
    name="rag_documents",
    embedding_function=embedding_function
)

def add_documents_to_chroma(documents):
    for i, doc in enumerate(documents):
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
    print("Type 'quit' to exit the chat.")
    
    # Add some sample documents to Chroma
    sample_docs = [
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris and was completed in 1889.",
        "Paris is known as the City of Light.",
        "The Louvre Museum in Paris houses the Mona Lisa painting.",
    ]
    add_documents_to_chroma(sample_docs)
    
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
        
        # Print retrieved documents
        print("\nRetrieved Documents:")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"{i}. {doc}")
        print()  # Add a blank line for better readability
        
        messages.append({"role": "user", "content": user_input})
        
        assistant_response = chat_with_gpt(messages, relevant_docs)
        print("Assistant:", assistant_response)
        print()  # Add a blank line for better readability
        
        messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()