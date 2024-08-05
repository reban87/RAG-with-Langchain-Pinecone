# @ IMPORT THE NECESSARY LIBRARIES
import uuid
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langsmith import traceable
from src.config.settings import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    TEMPERATURE,
    MODEL_NAME,
)
from src.storage.pinecone_utils import init_pinecone, get_or_create_index

# @ INITIATE THE OPENAI CLIENT
client = OpenAI(api_key=OPENAI_API_KEY)


class RagEngine:
    def __init__(self):
        pc = init_pinecone()
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # self.vectorstore = Pinecone.from_existing_index(
        #     PINECONE_INDEX_NAME, embedding=embeddings
        # )

        self.vectorstore = get_or_create_index(self.embeddings)

        print(f"Using OpenAI API Key: {OPENAI_API_KEY[:5]}...")
        llm = ChatOpenAI(
            temperature=TEMPERATURE,
            model_name=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
        )
        print(f"open_ai_key: {OPENAI_API_KEY}")
        template = """You are a Health Care Insurance Data Intrepretor bot. Use the following pieces of context to interpret the user's query. If the information can not be found in the context, just say "I don't know.
        Context: {context}
        Question: {question}
        Answer: """
        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 7}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        print(f"Debug: QA Chain input keys: {self.qa_chain.input_keys}")

    def process_documents(self, docs):
        """Process and store new documents from the vector stores."""
        try:
            self.vectorstore.add_documents(docs)
            print(f"Successfully added {len(docs)} documents to the vector store.")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")

    def clear_vectorstore(self):
        """Clear all documents from the vector store"""
        self.vectorstore.delete(delete_all=True)
        print("Vector store cleared.")

    @traceable(run_type="retriever")
    def retriever(self, query: str):
        """Retrieve relevant documents from the vector store"""
        return self.vectorstore.similarity_search(query, k=7)

    @traceable(metadata={"llm": "gpt-3.5-turbo"})
    def interpret_query(self, question, user_id=None):
        run_id = str(uuid.uuid4())
        print(f"Interpreting query: {question}")
        docs = self.retriever(question)
        print(f"Debug: Retrieved {len(docs)} documents")
        print(f"Debug: QA Chain input keys: {self.qa_chain.input_keys}")
        result = self.qa_chain.invoke(
            {"query": question, "input_documents": docs},
            config={"metadata": {"user_id": user_id} if user_id else {}},
        )
        answer = result["result"] if "result" in result else result["answer"]
        sources = [doc.page_content for doc in result["source_documents"]]
        return answer, sources, run_id

    def run_interactive_session(self):
        print("Start talking with the bot (type 'quit' to exit)")

        while True:
            question = input("User: ")
            if question.lower() == "quit":
                break
            elif question.lower() == "menu":
                print("Returning to main menu.")
                return "menu"

            user_id = input("Enter your user ID (or press enter to skip): ")
            try:
                answers, sources, run_id = self.interpret_query(
                    question, user_id if user_id else None
                )
                print(f"Answers: {answers}")
                # print(f"Sources: {sources}")
                print(f"\nRun ID: {run_id}")

                feedback = input("was this answer helpful? (y/n): ")
                if feedback.lower() == "y":
                    self.log_feedback(run_id, 1)
                elif feedback.lower() == "n":
                    self.log_feedback(run_id, 0)

                print("You want to coninue?[y/n]")
                if input() == "n":
                    break
                elif input == "y":
                    continue
                print("\n")
            except Exception as e:
                print(f"Error: {e}")

            print("\n")

    def log_feedback(self, run_id, score):
        from langsmith import Client

        ls_client = Client()
        ls_client.create_feedback(run_id=run_id, score=score, key="user_score")

    def get_qa_chain(self):
        return self.qa_chain
