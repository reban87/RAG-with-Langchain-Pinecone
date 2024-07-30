# @ IMPORTING NECESSARY LIBRARIES
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config.settings import DATA_DIR


def load_and_split_documents():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf")
    documents = loader.load()

    # @ INITIATE THE TEXT SPLITTER
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs
