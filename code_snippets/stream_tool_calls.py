from openai import OpenAI
import os
from dotenv import load_dotenv
from tools import TOOLS

load_dotenv()


client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

stream = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=[{
        "role" : "user",
        "content" : "what is the weather of pune and delhi"
    }],
    tools=TOOLS,
    stream=True
)


for chunk in stream:
    if not chunk.choices:
        continue

    piece = chunk.choices[0].delta
    print(piece.content,end="",flush=True)
    if piece.tool_calls:
        print(piece.tool_calls,end="",flush=True)

