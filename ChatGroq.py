from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())

# Prompt and model setup
system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
chat = ChatGroq(temperature=0, model_name="llama3-8b-8192")
chain = prompt | chat

# List to store message history
messages = []

def main():
    print("Chat with Language Model - LangChain")
    print("Type 'exit' to end the chat.")
    
    while True:
        # User input
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            break
        
        # Add the user's message to the history
        messages.append({"role": "user", "content": user_input})
        
        # Generate the model's response
        response_stream = chain.stream({"text": user_input})
        full_response = ""
        
        for partial_response in response_stream:
            full_response += str(partial_response.content)

        # Display the model's response
        print(f"Assistant: {full_response}")

        # Add the model's response to the history
        messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
