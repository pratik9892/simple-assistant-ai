from openai import OpenAI
from dotenv import load_dotenv
from os import getenv
import numpy

load_dotenv()

client = OpenAI(
    base_url=getenv("BASE_URL"),
    api_key=getenv("API_KEY")
)

def get_embedding(text):
    response = client.embeddings.create(
    model=getenv("EMBEDDING_MODEL"),
    input=text
    )
    return response.data[0].embedding

def cosine_similarity(A,B):
    dot_prod = numpy.dot(A,B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_prod / (norm_a * norm_b)

text1 = "Python is a programming language" 
text2 = "Python is used to write AI and Data related code"
text3 = "Honesty is the best policy"

# text 1 and 2 are similar and text 3 is different

embedding1 = get_embedding(text1)
embedding2 = get_embedding(text2)
embedding3 = get_embedding(text3)

cosine_sim = cosine_similarity(embedding1,embedding3)

print(cosine_sim)

# text 1 and 2 cosine similarity is 0.73 as they are similar or about same topic which is python programmning language
# text 1 and 3 cosine similarity is 0.39 as both have differnt meaning one is talking about python programmnig language and other is about Honesty
