import os
import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from tqdm import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS
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
CORS(app, resources={r"/*": {"origins": "*"}})

class PolicyChunk:
    def __init__(self, text, policy_name):
        self.text = text
        self.policy_name = self.format_policy_name(policy_name)

    @staticmethod
    def format_policy_name(policy_name):
        # Check if the policy name matches the pattern "Policy-XXXXXXXX"
        match = re.match(r'Policy-(\d{4})(\d{4})', policy_name)
        if match:
            # If it matches, format it as "XXXX/XXXX"
            return f"{match.group(1)}/{match.group(2)}"
        else:
            # If it doesn't match, return the original name
            return policy_name

class EmbeddingSearch:
    def __init__(self, embedding_file="embeddings.pkl"):
        self.embedding_file = embedding_file
        self.policy_chunks = []
        self.embeddings = []

    def load_embeddings(self):
        if os.path.exists(self.embedding_file):
            print("Loading pre-computed embeddings...")
            with open(self.embedding_file, 'rb') as f:
                self.policy_chunks, self.embeddings = pickle.load(f)
            return True
        else:
            print("No pre-computed embeddings found.")
            return False

    def save_embeddings(self):
        print("Saving embeddings...")
        with open(self.embedding_file, 'wb') as f:
            pickle.dump((self.policy_chunks, self.embeddings), f)

    def add_policy_chunks(self, policy_chunks):
        print("Encoding policy chunks...")
        new_embeddings = []
        for chunk in tqdm(policy_chunks, desc="Embedding policy chunks"):
            new_embeddings.append(model.encode(chunk.text))
        
        self.policy_chunks.extend(policy_chunks)
        self.embeddings.extend(new_embeddings)
        self.embeddings = np.array(self.embeddings)
        
        self.save_embeddings()

    def search(self, query, top_k=5):
        query_with_prompt = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = model.encode([query_with_prompt])[0]
        scores = np.dot(self.embeddings, query_embedding)
        top_results = np.argsort(scores)[::-1][:top_k]
        
        return [self.policy_chunks[i] for i in top_results]

# Initialize EmbeddingSearch
searcher = EmbeddingSearch()

def process_pdfs(pdf_dir):
    policy_chunks = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        filepath = os.path.join(pdf_dir, filename)
        policy_name = os.path.splitext(filename)[0]  # Use filename without extension as policy name
        try:
            elements = partition_pdf(filepath)
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                policy_chunks.append(PolicyChunk(chunk.text, policy_name))
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return policy_chunks

def add_policy_chunks_to_search(policy_chunks):
    searcher.add_policy_chunks(policy_chunks)

def retrieve_relevant_docs(query, n_results=40):
    return searcher.search(query, top_k=n_results)

def chat_with_gpt(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
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

    # Retrieve relevant policy chunks
    relevant_chunks = retrieve_relevant_docs(user_input)
    
    if not conversation_history:
        # Initialize with system message
        conversation_history = [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Use the relevant information provided to answer questions accurately. There may be information which is not related, use your best judgement to ignore it. Make sure to reference the source of the information, including the policy number in the format XXXX/XXXX, and don't make up information not given to you. Refernce the policy number of the information. Limit your response to 6 sentences or less. USE MARKDOWN FORMATTING TO MAKE THE INFORMATION EASIER TO READ WHENEVER POSSIBLE, SUCH AS USING LISTS, HEADINGS, BOLD, ITALICS, ETC."
            }
        ]
    
    # Attach relevant information to the user's query, including formatted policy names
    context = "Relevant information:\n" + "\n".join([f"From Policy {chunk.policy_name}: {chunk.text}" for chunk in relevant_chunks])
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

@app.route('/random_question', methods=['GET', 'OPTIONS'])
def get_random_question():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200
    
    random_question = random.choice(random_questions)
    return jsonify({"question": random_question})

if __name__ == "__main__":
    # Load existing embeddings or process PDFs
    if not searcher.load_embeddings():
        print("No existing embeddings found. Processing PDFs...")
        pdf_dir = 'pdf'  # Replace with your PDF directory path
        policy_chunks = process_pdfs(pdf_dir)
        add_policy_chunks_to_search(policy_chunks)
        print(f"Processed and added {len(policy_chunks)} policy chunks.")
    else:
        print(f"Loaded {len(searcher.policy_chunks)} policy chunks from existing embeddings.")

    app.run(host='0.0.0.0', port=80, debug=True)