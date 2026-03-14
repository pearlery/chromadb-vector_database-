from fastapi import FastAPI
from pydantic import BaseModel
from query import query_collection

app = FastAPI()

class Course(BaseModel):
    query: str
    

@app.post("/query/")
def query(course: Course):
    query_text = course.query
    result = query_collection(query_text)

    rank1 = result['ids'][0][0]
    rank2 = result['ids'][0][1]
    rank3 = result['ids'][0][2]

    return {"rank1":rank1,
            "rank2":rank2,
            "rank3":rank3}