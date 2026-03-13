from fastapi import FastAPI
from pydantic import BaseModel
from query import query_collection

app = FastAPI()

class Course(BaseModel):
    name: str
    description: str


@app.post("/query/")
def query(course: Course):
    query_text = course.description
    result = query_collection(query_text)

    return {"result": result}