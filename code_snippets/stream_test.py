from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = []

messages.append({
    "role" : "system",
    "content" : "Act as a humble and responsive person and give answer in 100-200 words"
})

while True:

    user_input = input("You : ")

    if(user_input.lower in ["exit" , "bye" , "close"]):
        break

    messages.append({
        "role" : "user",
        "content" : user_input
    })

    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        stream=True
    )

    print("Assistant : ", end="")
    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
         print(piece,end="",flush=True)

    print("")