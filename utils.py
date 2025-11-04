# utils.py - optional OpenAI wrapper
import os
try:
    import openai
except Exception:
    openai = None

def generate_insights_with_llm(summary_text: str, api_key: str) -> str:
    if openai is None:
        return 'OpenAI library not installed. Install `openai` to enable LLM insights.'
    openai.api_key = api_key
    prompt = (
        f"You are a data scientist assistant. Expand the following summary into 5 clear insights and recommended next steps:\\n\\n{summary_text}\\n\\nProvide short bullet points."
    )
    try:
        resp = openai.Completion.create(engine='text-davinci-003', prompt=prompt, max_tokens=300, temperature=0.2)
        text = resp.choices[0].text.strip()
        return text
    except Exception as e:
        return f'LLM call failed: {e}'
