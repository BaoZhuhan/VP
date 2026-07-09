"""Test single HL classification to debug 'unknown' issue."""
import json, os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

with open("data/dialogues/P001_standard.json") as f:
    dialogue = json.load(f)

utterances = "\n\n".join(
    [f"Patient: {t['content']}" for t in dialogue["turns"] if t["role"] == "patient"]
)

prompt = f"""Read these patient utterances from a clinical consultation and determine the patient's health literacy level (low, medium, or high).

PATIENT UTTERANCES:
{utterances}

Respond with ONLY one word: "low", "medium", or "high". Do not explain."""

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=20,
)
text = resp.choices[0].message.content
print(f"RAW: {repr(text)}")
for word in ["low", "medium", "high"]:
    if word in text.lower():
        print(f"MATCH: {word}")
