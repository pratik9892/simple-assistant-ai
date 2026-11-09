import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy
import json

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")
VECTOR_STORE = os.getenv("VECTOR_STORE_FILE")



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


# create chunks from loaded documets

def create_chunks():
    documents = load_docs()
    chunks = []

    for doc in documents:
        chunk_id = 0
        for part in doc["text"].split("\n\n"):
            part = part.strip()
            if part:
                chunks.append(
                    {
                        "source" : doc["source"],
                        "chunk_id" : chunk_id,
                        "text" : part
                    } 
                )
                chunk_id += 1

    return chunks

# it is used to save the chunk vector into file

def save_vector_store(chunks):
    with open(VECTOR_STORE,"w",encoding="utf-8") as f:
        json.dump(chunks,f,indent=4)


# this method will help to laod the chunks in vectore stroe file

def load_vectore_store():
    with open(VECTOR_STORE,"r",encoding="utf-8") as f:
        return json.load(f) #returns the chunk 


# add embeddings in the chunks

def add_embedding(chunks):
    for chunk in chunks:
        vector = get_embeddings(chunk["text"])
        chunk["vector"] = vector

    return chunks 



def load_chunks():
    if os.path.exists(VECTOR_STORE):
      chunks = load_vectore_store()
      print(f"{len(chunks)} Chunks loaded from vector store file - {VECTOR_STORE}")
    else:
      chunks = create_chunks()
      chunks= add_embedding(chunks)
      save_vector_store(chunks)
      print(f"Created and embedded {len(chunks)} into vector store file - {VECTOR_STORE}")

    return chunks  

chunks = load_chunks()

# search with query, chunks and top_k

def search(query, top_k=3):
    query_vec = get_embeddings(query)
    results = []

    for chunk in chunks:
        score = cosine_similarity(chunk["vector"],query_vec)

        results.append({
            "score" : float(score),
            "source" : chunk["source"],
            "text" : chunk["text"],
            "chunk_id" : chunk["chunk_id"], 
            "size" : len(chunk["text"])
        })


    # sort the results

    results.sort(key=lambda rec:rec["score"], reverse=True)
    return results[:top_k]