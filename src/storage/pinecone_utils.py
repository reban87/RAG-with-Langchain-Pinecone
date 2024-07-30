from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec
from src.config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


def init_pinecone():
    return PineconeClient(api_key=PINECONE_API_KEY)


def get_or_create_index(embeddings):
    pc = init_pinecone()
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print("I am here to create a new index now")
        pc.create_index(
            PINECONE_INDEX_NAME,
            dimension=len(embeddings.embed_query("test")),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return Pinecone.from_existing_index(PINECONE_INDEX_NAME, embedding=embeddings)
