from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config.settings import EMBEDDING_MODEL
from openai import OpenAI

client = OpenAI()

# def get_embeddings():
#     model_kwargs = {"device": "cpu"}
#     encode_kwargs = {"normalize_embeddings": False}
#     return HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL,
#         model_kwargs=model_kwargs,
#         encode_kwargs=encode_kwargs,
#     )


def get_embedding(text, model="text-embedding-3-small"):

   text = text.replace("\n", " ")

   return client.embeddings.create(input = [text], model=model).data[0].embedding
 