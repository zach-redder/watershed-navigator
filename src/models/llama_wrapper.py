import os
from dotenv import load_dotenv
import requests

load_dotenv()

API_KEY = os.getenv("HUGGINGFACE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def ask_llama(prompt: str, max_new_tokens=512) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"❌ API error: {response.status_code} - {response.text}")

    output = response.json()
    if isinstance(output, list):
        return output[0]["generated_text"].replace(prompt, "").strip()
    elif "error" in output:
        raise RuntimeError(f"❌ Model error: {output['error']}")
    return output
