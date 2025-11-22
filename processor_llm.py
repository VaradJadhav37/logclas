# processor_llm.py  (safe lazy LLM call)
import os
import re
import traceback

_groq_client = None
_MODEL_NAME = "llama-3.1-8b-instant"  # keep your configured model

def _ensure_groq():
    global _groq_client
    if _groq_client is not None:
        return
    try:
        from dotenv import load_dotenv
        from groq import Groq
        load_dotenv()
        _groq_client = Groq()
    except Exception as e:
        print("Warning: failed to initialize Groq LLM client:", e)
        print(traceback.format_exc())
        _groq_client = None

def classify_with_llm(log_msg: str) -> str:
    _ensure_groq()
    if _groq_client is None:
        return "Unclassified"

    prompt = f'''Classify the log message into one of these categories: 
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category, use "Unclassified".
    Put the category inside <category> </category> tags. 
    Log message: {log_msg}'''

    try:
        chat_completion = _groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=_MODEL_NAME,
            temperature=0.0
        )
        content = chat_completion.choices[0].message.content
        match = re.search(r'<category>(.*)<\/category>', content, flags=re.DOTALL)
        if match:
            category = match.group(1).strip()
            return category
        return "Unclassified"
    except Exception as e:
        print("Error calling LLM:", e)
        print(traceback.format_exc())
        return "Unclassified"
