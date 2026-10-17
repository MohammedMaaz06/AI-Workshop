import json
import re
import urllib.request

question = input("User: ")

url = "http://localhost:11434/api/chat"
data = {
    "model": "qwen3.5:0.8b",
    "messages": [
        {
            "role": "system",
            "content": "Do not use bullet points or numbering.",
        },
        {"role": "user", "content": question},
    ],
    "options": {
        "temperature": 300,
    },
    "stream": False,
}

req = urllib.request.Request(
    url,
    data=json.dumps(data).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        raw_content = result.get("message", {}).get("content", "")

        clean_answer = re.sub(
            r"<think>.*?</think>", "", raw_content, flags=re.DOTALL
        ).strip()

        print("\nAI:", clean_answer if clean_answer else raw_content)
except Exception as e:
    print(f"\nError connecting to Ollama: {e}")
