from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings 
from config.settings import (
    CHROMA_DB_PATH,
    OPENAI_API_KEY
)
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations

# def init_chroma():
#     return Chroma(persist_directory=CHROMA_DB_PATH)


def save_to_chroma(chunks):
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

  # Create a new Chroma database from the documents using OpenAI embeddings
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        persist_directory=CHROMA_DB_PATH
    )

    # Persist the database to disk
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_DB_PATH}.")


def init_chroma_db(embedding_function):
    return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

# if __name__ == "__main__":
#     # This allows running this script directly for testing
#     embeddings = HuggingFaceEmbeddings()
#     vectorstore = get_or_create_index(embeddings)
#     print(f"Vectorstore initialized with index: default_index")
