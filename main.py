import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def chat_with_gpt(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    print("Welcome to the GPT-4o mini Chatbot!")
    print("Type 'quit' to exit the chat.")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        messages.append({"role": "user", "content": user_input})
        
        assistant_response = chat_with_gpt(messages)
        print("Assistant:", assistant_response)
        
        messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()