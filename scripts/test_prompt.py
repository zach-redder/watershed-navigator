import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.llama_wrapper import ask_llama

prompt = """You are a helpful assistant for water scientists.

Context:
The Huron River watershed spans over 900 square miles and drains into Lake Erie. It includes forested land, urban areas, and farmland.

Question:
What are some environmental concerns for this watershed?

Answer:"""

response = ask_llama(prompt)
print("AI Response:\n")
print(response)
