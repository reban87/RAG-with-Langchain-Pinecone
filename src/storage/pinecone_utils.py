from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec
from src.config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_ENVIRONMENT,
)


def init_pinecone():
    return PineconeClient(api_key=PINECONE_API_KEY)


def get_or_create_index(
    embeddings, index_name=PINECONE_INDEX_NAME, environment=PINECONE_ENVIRONMENT
):
    pc = init_pinecone()
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=len(embeddings.embed_query("test")),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=environment),
        )
    return Pinecone.from_existing_index(index_name, embedding=embeddings)


if __name__ == "__main__":
    # This allows  to run this script directly for testing
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings()
    vectorstore = get_or_create_index(embeddings)
    print(f"Vectorstore initialized with index: {PINECONE_INDEX_NAME}")
