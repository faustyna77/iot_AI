from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="mistralai/mistral-7b-instruct",  # lub np. "mistralai/mistral-7b-instruct"
)
message = """
Dane z czujników:
- napięcie: 12.3V
- światło: 800lx
- prąd ładowania: 2.4A
- prąd rozładowania: 0.6A

Czy należy ładować akumulator, rozładowywać czy zostawić w spoczynku?
"""

response = llm([HumanMessage(content=message)])
print(response.content)
