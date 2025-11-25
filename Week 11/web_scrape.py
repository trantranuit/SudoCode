import os
import requests
from bs4 import BeautifulSoup

# =============================
# CONFIG — điền API KEY OpenAI
# =============================
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_MODEL"] = "gpt-4o-mini"

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL")
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"


# =============================
# 1. Web Scraper
# =============================
def scrape_url(url: str) -> str:
    print(f"Đang scrape: {url}")
    resp = requests.get(url, timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator="\n")
    cleaned = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    print(f"Đã thu được {len(cleaned)} ký tự nội dung.")
    return cleaned


# =============================
# 2. OpenAI Chat API
# =============================
def call_llm(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 600
    }

    resp = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=40)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"]


# =============================
# 3. CLI hỏi đáp
# =============================
def qa_cli(context_text: str):
    print("\n==============================")
    print("🤖 CLI Q&A dựa trên nội dung trang web")
    print("Nhập 'exit' để thoát")
    print("==============================\n")

    while True:
        user_q = input("Câu hỏi: ")
        if user_q.lower().strip() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        full_prompt = f"""
Dưới đây là nội dung đã scrape từ trang web:

-----------------
{context_text}
-----------------

Hãy trả lời câu hỏi dựa trên nội dung trên:

{user_q}
"""

        answer = call_llm(full_prompt)
        print("\n💡 Trả lời:")
        print(answer)
        print("\n--------------------------------------\n")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    url = input("Nhập URL cần scrape: ").strip()
    context = scrape_url(url)
    qa_cli(context)
