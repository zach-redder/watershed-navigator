def format_prompt(context: str, question: str) -> str:
    return f"""You are a helpful assistant for water scientists and environmental modelers.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
