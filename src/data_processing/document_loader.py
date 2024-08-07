# @ IMPORTING NECESSARY LIBRARIES
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import DATA_DIR
import logging


# @ SET LOGGING LEVEL
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_split_documents(
    chunk_size=1000, chunk_overlap=200, file_pattern="**/*.pdf"
):
    try:
        # Load documents
        logger.info(f"Loading documents from {DATA_DIR}")
        loader = DirectoryLoader(DATA_DIR, glob=file_pattern)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        # Split documents
        logger.info("Splitting documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"Created {len(docs)} document chunks")

        return docs

    except Exception as e:
        logger.error(f"Error in load_and_split_documents: {e}")
        raise

def load_documents(
    file_pattern="**/*.pdf"
):
    try:
        # Load documents
        logger.info(f"Loading documents from {DATA_DIR}")
        loader = DirectoryLoader(DATA_DIR, glob=file_pattern , show_progress=True)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        return documents

    except Exception as e:
        logger.error(f"Error in load_and_split_documents: {e}")
        raise


def load_and_split_multiple_file_types():
    pdf_docs = load_and_split_documents(file_pattern="**/*.pdf")
    txt_docs = load_and_split_documents(file_pattern="**/*.txt")
    # Add more file types as needed
    return pdf_docs + txt_docs  # Combine all document chunks


# if __name__ == "__main__":
    # This allows you to run this script directly for testing
    # docs = load_and_split_documents()
    # print(f"Total chunks: {len(docs)}")
