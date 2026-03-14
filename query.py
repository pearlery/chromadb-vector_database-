import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings

CHROMA_DB_PATH = 'chroma_db'
COLLECTION_NAME = "my_collection" #database name 

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name="my_collection")

def query_collection(query:str):
    top_k = 3

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    return results

text = "วิชาที่เรียนเกี่ยวกับระบบคอมพิวเตอร์"

result = query_collection(query=text)

print(result)

rank1 = result['ids'][0][0]
print(rank1)

rank2 = result['ids'][0][1]
print(rank2)