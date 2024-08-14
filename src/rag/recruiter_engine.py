# @ IMPORT THE NECESSARY LIBRARIES
import uuid
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.load import loads

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langsmith import traceable
from config.settings import (
    OPENAI_API_KEY,
    TEMPERATURE,
    MODEL_NAME,
)
from storage.pinecone_utils import init_pinecone, get_or_create_index
from langchain.embeddings.openai import OpenAIEmbeddings
from datetime import date


# @ INITIATE THE OPENAI CLIENT
client = OpenAI(api_key=OPENAI_API_KEY)
today = date.today()


class RecruiterRagEngine:
    def __init__(self):
        pc = init_pinecone()
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        self.vectorstore = get_or_create_index(self.embeddings)

        print(f"Using OpenAI API Key: {OPENAI_API_KEY[:5]}...")
        llm = ChatOpenAI(
            temperature=TEMPERATURE,
            model_name=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
        )
        print(f"open_ai_key: {OPENAI_API_KEY}")

        resume_schema = {
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "location": {"type": "string"},
                "summary": {"type": "string"},
                "skills": [{"type": "str"}],
                "total_years_of_experience": {"type": "string"},
                "education": {"type": "string"},
                "job_description": {"type": "string"},
            }
        }

        query_classifier_template = """
            You are an AI assistant for a talent acquisition team. Your task is to classify the given query into one of two categories:
            1. Candidate Sorting: Queries that ask to rank, sort, or compare multiple candidates based on certain criteria.
            2. Specific Query: Questions about specific experiences, or individual candidates.
            
            if a question has a job description classify it as  candidate sorting
            
            Question: {question}

            Please respond with either "Candidate Sorting" or "Specific Query".

            query_type:

            """

        # query_additional_questions_template = """ Generate 3 different versions of the following query, each focusing on a different aspect or interpretation:

        #         Original query: {question}

        #         1.
        #         2.
        #         3.
        #         """

        query_additional_questions_template = """You are an Ai assistant for a talent acquisition team.Your task is to generate questions according to the query provided so that all the important criterias are covered while filtering resumes.
            Start the questions with 'Find candidates with'
 
 
            query: {question}

            Here's the example for your reference:
            query: Experience in machine learning and cloud computing

            response:

            Find candidates with a strong background in machine learning algorithms and their implementation.
            Find candidates with hands-on experience in cloud platforms like AWS, Google Cloud, or Azure for deploying ML models.
 
            Your response should be in the following format:
            1.[Question1]
            2.[Question2]
            ...
  """

        sort_candidates_template = f"""
            You are an AI assistant for a talent acquisition team. Your task is to sort and rank candidates based on their resumes and the given query.
            Todays date is {today}.

            Questions: {{question}}

            Relevant Resume Information:
            {{context}}

            Task:
            Please analyze each resume, compare it with the questions, and treat the questions as a checklist. For each candidate, provide a true or false response for each question, and then generate a compatibility percentage. Afterward, sort and rank the candidates based on their compatibility percentage.

            Output:

                - Ensure each candidate appears only once in the final list.
                - Provide a sorted list of candidates from most suitable to least suitable.
                - If two candidates have the same compatibility percentage, rank them based on the relevance and strength of their experience.
                - Ensure no duplicate candidates are listed in the final output.
               
            Format:

                1. [Candidate Name] [Compatibility Percentage]: [Brief explanation]


                2. [Candidate Name] [Compatibility Percentage]: [Brief explanation]
                ...     

                - If fewer candidates match than expected, list only those who match.
                - If no candidates meet the criteria, return 'No candidates found.
                
            Ensure no duplicate candidates are listed in the final output.

            AI Response:
                """

        specific_query_template = f"""
            You are an AI assistant for a talent acquisition team. Your task is to answer specific questions about candidates or skills based on their resumes.Do not give details that are not in their own resumes.
            The Skills mentioned in the question should be an extact match if it is not listed as optional or nice to have.Skills can be in groups divided by commas in which at least one has to be a exact match.
            Todays date is {today}.

            Question: {{question}}

            Relevant Resume Information:
            {{context}}

            Please provide a detailed and professional response, suitable for use in HR decision-making. 
            If the query is about a specific skill, focus on candidates who possess that skill and mention where, when and in which project they have used it. 
            If it's about a particular candidate, provide detailed information about that candidate's qualifications and experiences.
            Do not include anything that are not in their own resume.

            Your response should be structured and easy to read. Use bullet points or numbering when appropriate.

            Response:
            """

        self.query_classifier_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=query_classifier_template,
                input_variables=["question"],
            ),
            verbose=True,
        )

        self.query_addtional_questions_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                # template=query_classifier_template,
                template=query_additional_questions_template,
                input_variables=["question"],
            ),
            verbose=True,
        )

        self.sort_candidates_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.8, "k": 3},
            ),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=sort_candidates_template,
                    input_variables=["question", "context"],
                ),
            },
            verbose=True,
        )

        self.specific_query_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.8, "k": 5},
            ),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=specific_query_template,
                    input_variables=["question", "context"],
                ),
            },
            verbose=True,
        )

    def process_documents(self, docs):
        """Process and store new documents from the vector stores."""
        try:
            self.vectorstore.add_documents(docs)
            print(f"Successfully added {len(docs)} documents to the vector store.")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")

    def vector_retrieval(self, query, query_type):
        if query_type == "Candidate Sorting":
            results = self.sort_candidates_chain.retriever.get_relevant_documents(query)
        elif query_type == "Specific Query":
            results = self.specific_query_chain.retriever.get_relevant_documents(query)
        else:
            results = self.specific_query_chain.retriever.get_relevant_documents(query)

        return {
            doc.metadata.get("id", f"doc_{i}"): {
                "score": getattr(doc, "score", 0.0),
                "content": doc.page_content,
            }
            for i, doc in enumerate(results)
        }

    def clear_vectorstore(self):
        """Clear all documents from the vector store"""
        self.vectorstore.delete(delete_all=True)
        print("Vector store cleared.")

    @traceable(metadata={"llm": "gpt-3.5-turbo"})
    def interpret_query(self, question, user_id=None):
        run_id = str(uuid.uuid4())
        print(f"Interpreting query: {question}")
        if question:
            # classify the question first
            query_type = self.query_classifier_chain.invoke(question)["text"]
            print(f"query_type:{query_type}")
            # retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            # results = self.vectorstore.similarity_search(question,k=10)
            # print(f"Query: {question}")
            # print(f"Results: {results}")
            # print(f"Result count: {len(results)}")
            additional_questions = self.query_addtional_questions_chain.invoke(question)
            all_questions = [question] + [
                q for q in additional_questions["text"].splitlines() if q.strip()
            ]  # Original + 3 additional questions

            # @ Retrieve the documents based upon the query
            search_results_dict = {}
            for i, q in enumerate(all_questions):
                search_results_dict[f"query_{i}"] = self.vector_retrieval(q, query_type)

            # Apply Reciprocal Rank Fusion to rerank the documents
            reranked_results = self.reciprocal_rank_fusion(search_results_dict)
            print(f"rerannked_results: {reranked_results}")

            top_docs = list(reranked_results)[:5]

            # Generate the final answer based on reranked results
            # context = "\n".join([doc.page_content for doc in search_results])
            context = "\n".join(
                [
                    search_results_dict[query][doc_id]["content"]
                    for query in search_results_dict
                    for doc_id in top_docs
                    if doc_id in search_results_dict[query]
                ]
            )

            # @ Generate the final answer based on reranked results
            if query_type == "Candidate Sorting":
                result = self.sort_candidates_chain.invoke(
                    {"query": question, "context": context}
                )
            elif query_type == "Specific Query":
                result = self.specific_query_chain.invoke(
                    {"query": question, "context": context}
                )
            else:
                # Fallback to specific query if classification fails
                result = self.specific_query_chain.invoke(
                    {"query": question, "context": context}
                )

            answer = (
                result
                if isinstance(result, str)
                else result.get("result") or result.get("answer")
            )
            # sources = (
            #     [doc.page_content for doc in result["source_documents"]]
            #     if isinstance(result, dict)
            #     else []
            # )
            # return answer, sources, run_id
            return answer, run_id

        else:
            print("No question provided")
            return None, None

    def log_feedback(self, run_id, score):
        from langsmith import Client

        ls_client = Client()
        ls_client.create_feedback(run_id=run_id, score=score, key="user_score")

    def get_qa_chain(self):
        return self.qa_chain

    def reciprocal_rank_fusion(self, search_results_dict: dict, k=60):
        fused_scores = {}
        print("Initial individual search result ranks:")
        for query, doc_scores in search_results_dict.items():
            print(f"For query '{query}': {doc_scores}")

        for query, doc_scores in search_results_dict.items():
            for rank, (doc, details) in enumerate(
                sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            ):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                previous_score = fused_scores[doc]
                fused_scores[doc] += 1 / (rank + k)
                print(
                    f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'"
                )

        reranked_results = dict(
            sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        )
        print("Final reranked results:", reranked_results)
        return reranked_results
