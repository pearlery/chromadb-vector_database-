import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


class tranformer:
    def __init__(self):
        self.df = pd.read_csv('training_dataset 2.csv') 
        self.model = None 
        self.corpus = self.df['description'].tolist()
        self.embedding = None 
        self.d = 768
        self.flat_index = None 
    
        
    def load_model(self):
        try:
            self.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            print('Model loaded.')
            return
        except Exception as e:
            print(f'Error to load model:{e}')
            return
        
    def embedding_corpus(self):
        self.embedding = self.model.encode(self.corpus)
        print('Corpus encoded.')
        return

    def flat_index_init(self): 
        self.flat_index = faiss.IndexFlatIP(self.d)
        self.flat_index.add(self.embedding)
        print('init flat index.')
        return
    
    def query_flat_index(self,query_text):
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)
        D, I = self.flat_index.search(query_embedding, 10) 
        result_courses = [self.corpus[idx] for idx in I.flatten()]
        return result_courses
    

    def query_embedding(self,query_text):
        query_embedding = self.model.encode([query_text],normalize_embeddings=True)
        print(query_embedding.shape)
        return query_embedding 
    
    def run(self):
        return

        
