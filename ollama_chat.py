from ollama import Client

# 1. Initialize the Ollama client
client = Client(host="http://localhost:11434")

# 2. Store the conversation history
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Keep your answers simple, clear, and informative. "
            "Do not use bullet points or numbering."
        ),
    }
]

print("Welcome to Ollama AI Chatbot!")
print("Type 'exit' to stop chatting.\n")

# 3. Main chat loop
while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    # Append user message to history
    messages.append({"role": "user", "content": user_input})

    # Call the model (change 'llama3' to your installed model)
    response = client.chat(model="llama3", messages=messages)

    # Extract assistant reply
    assistant_reply = response["message"]["content"]
    print(f"\nAssistant: {assistant_reply}\n")

    # Append assistant reply to preserve conversation context
    messages.append({"role": "assistant", "content": assistant_reply})
