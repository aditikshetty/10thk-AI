import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "What is 2 + 2?"
        }
    ]
)

print(response["message"]["content"])