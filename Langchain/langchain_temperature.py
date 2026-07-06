from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatOpenAI(model = "gpt-5", temperature=1.2)

result = model.invoke("Write a poem on Life in 5 lines.")

print(result.content)