import os
from dotenv import load_dotenv

print(f"Current working directory: {os.getcwd()}")
print(f".env file path: {os.path.join(os.getcwd(), '.env')}")

load_dotenv()

# @ PINECONE SETTINGS

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")


# @ OPENAI SETTINGS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEMPERATURE = os.getenv("TEMPERATURE")
MODEL_NAME = os.getenv("MODEL_NAME")

# @ MODEL SETTINGS
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# @ DATA SETTINGS
Base_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(Base_DIR, "data", "tmp")


# @ LANGSMITH API KEY
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")


if __name__ == "__main__":
    print(PINECONE_API_KEY)
    print(PINECONE_INDEX_NAME)
    print(PINECONE_ENVIRONMENT)
    print(OPENAI_API_KEY)
