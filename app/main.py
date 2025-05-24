from fastapi import FastAPI, HTTPException,Form
from pydantic import BaseModel
from app.webpage_reader import load_url_to_vectorstore ,ask_question


app = FastAPI()

# Pydantic model for POST body
class UrlRequest(BaseModel):
    url: str

@app.post("/upload")
async def ingest_url(request: UrlRequest):
    try:
        await load_url_to_vectorstore(request.url)
        return {"message": f"Successfully ingested: {request.url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest URL: {str(e)}")

@app.post("/ask")
def ask_url(query: str = Form(...)):
    answer = ask_question(query)
    return {"answer": answer}