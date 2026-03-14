from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader


query = "การพยาบาล"

courses = [
    "การดูแลแบบประคับประคอง",
    "จิตวิทยาประยุกต์ในการทำงาน เพื่อความสำเร็จ",
    "บริการสารสนเทศสำหรับองค์กรดิจิทัล",
]

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


# 2. Calculate embeddings by calling model.encode()
query_embeddings = model.encode(query, normalize_embeddings=True)
print(query_embeddings.shape)

courses_embeddings = model.encode(courses, normalize_embeddings=True)
print(courses_embeddings.shape)


# 3. Calculate the embedding similarities
similarities = model.similarity(query_embeddings, courses_embeddings)
print(similarities)