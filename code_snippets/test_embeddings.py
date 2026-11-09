from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()

client = OpenAI(
    base_url=getenv("BASE_URL"),
    api_key=getenv("API_KEY")
)

response = client.embeddings.create(
    model=getenv("EMBEDDING_MODEL"),
    input="My name is pratik"
)

print(response.data[0].embedding)