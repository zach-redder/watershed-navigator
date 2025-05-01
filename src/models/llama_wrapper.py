import requests
import json

def ask_llama(prompt: str, max_new_tokens=512) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_new_tokens}
        },
        stream=True
    )

    if response.status_code != 200:
        raise RuntimeError(f"❌ API error: {response.status_code} - {response.text}")

    final_response = ""

    for line in response.iter_lines():
        if line:
            part = line.decode('utf-8')
            json_data = json.loads(part)
            if 'response' in json_data:
                final_response += json_data['response']

    return final_response.strip()