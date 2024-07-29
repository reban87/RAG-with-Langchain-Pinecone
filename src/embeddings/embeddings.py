from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config.settings import EMBEDDING_MODEL, PINECONE_DIMENSION


def get_embeddings():
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
