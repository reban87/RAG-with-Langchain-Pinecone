# Recruitment Bot

## Description
This project is a Retrieval-Augmented Generation (RAG) system designed for recruitment. It uses advanced natural language processing techniques to answer queries based on ingested documents, providing accurate and context-aware responses based upon resumes.

## Features
- Document ingestion and processing (PDF support)
- Vector storage using Pinecone
- Query interpretation using OpenAI's language models
- Interactive CLI interface
- RESTful API using FastAPI and LangServe
- Monitoring and feedback collection with LangSmith

## Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key
- LangSmith API key (for monitoring)

## Installation

1. Clone the repository
2. Create a virtual environment
3. Set up environment variables:
Create a `.env` file in the project root and add the following:
```sample .env```

        PINECONE_API_KEY=your_pinecone_api_key  
        PINECONE_INDEX_NAME=your_index_name  
        PINECONE_ENVIRONMENT=your_pinecone_environment  
        OPENAI_API_KEY=your_openai_api_key  
        TEMPERATURE=0.7  
        MODEL_NAME=gpt-3.5-turbo  
        LANGSMITH_API_KEY=your_langsmith_api_key  
## Usage

### CLI Mode
1. Place your PDF documents in the `data/tmp` folder.
2. Run the CLI application:
```python3 src/main.py``` if it doesn't work then ```python3 -m src.main```
3. API Mode:
```python3 src/api/server.py``` if it doesn't work then ```python3 -m src.api.server```
