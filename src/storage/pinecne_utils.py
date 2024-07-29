from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from src.config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    PINECONE_ENVIRONMENT,
)


# @ TODO: CREATE PINECONE ENVIRONEMENT FROM THE PINECONE SERVICE
def init_pinecone():
    return PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


def get_or_create_index(embeddings):
    pc = init_pinecone()

    if PINECONE_INDEX_NAME not in pc.list.indexes().name():
        pc.create_index(
            PINECONE_INDEX_NAME, dimension=PINECONE_DIMENSION, metric="cosine"
        )
    return Pinecone.from_existing_index(PINECONE_INDEX_NAME, embedding=embeddings)
