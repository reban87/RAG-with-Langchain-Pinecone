# @ IMPORT THE NECESSARY LIBRARIES
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from src.config.settings import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_DIMENSION,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    TEMPERATURE,
    MODEL_NAME,
)
from src.storage.pinecne_utils import init_pinecone

# @ INITIATE THE OPENAI CLIENT
client = OpenAI(api_key=OPENAI_API_KEY)


class RagEngine:
    def __init__(self) -> None:
        pc = init_pinecone()
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        self.vectorstore = Pinecone.from_existing_index(
            PINECONE_INDEX_NAME, embedding=embeddings
        )

        llm = ChatOpenAI(
            temperature=TEMPERATURE,
            model_name=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
        )

        template = """You are a Health Care Insurance Data Intrepretor bot. Use the following pieces of context to interpret the user's query. If the information can not be found in the context, just say "I don't know.
        Context: {context}
        Question: {question}
        Answer: """
        prompt = PromptTemplate.from_template(
            template=template, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 7}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def interpret_query(self, query):
        docs = retriever(query)
        result = self.qa_chain.invoke(
            {
                "question": query,
            }
        )

        answers = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]

        return answers, sources

    def log_feedback(self, score):
        from langsmith import Client

        ls_client = Client()
        ls_client.create_feedback(
            score=score,
        )

    def get_qa_chain(self):
        return self.qa_chain
