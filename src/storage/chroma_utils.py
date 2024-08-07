from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import (
    CHROMA_DB_PATH,  # Replace this with your Chroma DB path or configuration
)


def init_chroma():
    return Chroma(persist_directory=CHROMA_DB_PATH)


def get_or_create_index(embeddings, index_name="default_index"):
    chroma = init_chroma()
    
    # Check if the index exists, and if not, create it
    if index_name not in chroma.list_indexes():
        chroma.create_index(name=index_name, dimension=len(embeddings.embed_query("test")))

    return chroma


if __name__ == "__main__":
    # This allows running this script directly for testing
    embeddings = HuggingFaceEmbeddings()
    vectorstore = get_or_create_index(embeddings)
    print(f"Vectorstore initialized with index: default_index")
