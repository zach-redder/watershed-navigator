from src.rag.prompt import format_prompt
from src.models.llama_wrapper import ask_llama
from src.rag.retrieve import get_top_k_docs

def answer_question(question: str) -> str:
    docs = get_top_k_docs(question, k=3)
    context = "\n\n".join([doc['text'] for doc in docs])
    prompt = format_prompt(context, question)
    return ask_llama(prompt)