from fastapi import FastAPI
from langserve import add_routes
from src.rag.engine import RagEngine
import uvicorn


app = FastAPI(
    title="Health Care Insurance Data Interpretor",
    version="1.0.",
    description="Health Care Insurance Data Interpretor",
)

# @ INITIALIZE THE ENGINE
engine = RagEngine()

# @ ADD ROUTES FOR THE RAG ENGINE CHAIN
add_routes(
    app,
    engine.get_qa_chain(),
    path="/health-care-agent",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
