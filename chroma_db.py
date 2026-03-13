import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings

import pandas as pd
import numpy as np

COLLECTION_NAME = "my_collection" #database name 
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2" #transformer embedding model
CHROMA_DB_PATH = "chroma_db" #databas path 


df = pd.read_csv("training_dataset 2.csv") #load data from csv file
print(df.columns) #print first 5 rows of the dataframe to check if data is loaded correctly


ids = df.index.astype(str).tolist() #create list of ids from dataframe index
documents = df['description'].tolist() #create list of documents from dataframe column 'text'
category_id = df['category_id'].tolist() #create list of category ids from dataframe column 'category_id'
category_name = df['category_name'].tolist() #create list of category names from dataframe column 'category_name'
source_utl = df['source_url'].tolist() #create list of source urls from dataframe column 'source_url'


#Create client(Database)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

#create embedding model 
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

#Create collection(table)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func, # embbeding function from sentence-transformer
    metadata={"hnsw:space": "cosine"}, # use cosine similarity to find closeness result. You can choose another include l2, ip.
)

#add data to collection
collection.add(documents=documents, 
               ids=ids, 
               metadatas=[{"category_id": cat_id, 
                           "category_name": cat_name, 
                           "source_url": url} for cat_id, cat_name, url in zip(category_id, category_name, source_utl)])


#print number of documents in the collection
print(f"Number of documents in the collection: {collection.count()}")

#print first 5 documents in the collection to check if data is added correctly
print(collection.peek())

def query_collection(query:str):
    top_k = 3

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    return results

text = "วิชาที่เรียนเกี่ยวกับระบบคอมพิวเตอร์"

query = query_collection(query=text)
print(query)

