import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")


# embedding logic

def get_embeddings(text):
    res = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=text
    )

    return res.data[0].embedding


# cosine similarity

def cosine_similarity(A,B):
    dot_prod = numpy.dot(A,B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_prod / (norm_a * norm_b)

#load the documents

def load_docs(folder=KB_FOLDER):
    documents = []

    for name in sorted(os.listdir(folder)):
        if not name.endswith(".txt"):
            continue

        path = os.path.join(folder,name)

        with open(path , "r" , encoding="utf-8") as f:
           documents.append({
            "source" : name,
            "text" : f.read()
           })

    return documents

# add embeddings in the documents

def add_embedding(documents):
    for doc in documents:
        vector = get_embeddings(doc["text"])
        doc["vector"] = vector

    return documents 

# search with query, documents and top_k

def search(query, documents, top_k=3):
    query_vec = get_embeddings(query)
    results = []

    for doc in documents:
        score = cosine_similarity(doc["vector"],query_vec)

        results.append({
            "score" : float(score),
            "source" : doc["source"],
            "text" : doc["text"],
            "size" : len(doc["text"])
        })


    # sort the results

    results.sort(key=lambda rec:rec["score"], reverse=True)
    return results[:top_k]


#test

def test(user_query, documents, top_k=3):
    print(f"\n User Query : {user_query}")
    results = search(user_query,documents,top_k)
    # return top-k results

    for r in results:
        print(f" {r["score"]:.2f} - {r['source']} - {r["size"]} characters")


documents = load_docs()
print(documents[0])
# doc_embeddings = add_embedding(documents)

# # we have embedded our documnets 

# user_query = "is dental covered"
# test(user_query,doc_embeddings,3)
 
