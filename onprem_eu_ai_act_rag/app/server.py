from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_community.llms import VLLMOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser

from pydantic import BaseModel, Field, ConfigDict, field_validator

from langchain.schema.runnable import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing import Any, List, Union
import pydantic


url = "http://localhost:6333"

app = FastAPI()
# llm = VLLMOpenAI(
#     openai_api_key="wiz-andromeda-001",
#     openai_api_base="http://192.168.2.42:8000/v1",
#     model_name="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
# )

# RAG_PROMPT_TEMPLATE = """\
# <|start_header_id|>system<|end_header_id|>
# You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

# <|start_header_id|>user<|end_header_id|>
# User Query:
# {query}

# Context:
# {context}<|eot_id|>

# <|start_header_id|>assistant<|end_header_id|>
# """

llm = ChatOllama(
    model='llama3.2:1b',
    temperature=0
)

RAG_PROMPT_TEMPLATE = """\
Answer the questions based on the context provided. If the context doesn't have the answer, say that you
don't know the answer.

User Query:
{query}

Context:
{context}
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)

client = QdrantClient(url=url)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="EU_AI_Act",
    embedding=embeddings,
)

retriever = vector_store.as_retriever()

lcel_rag_chain = {"context": itemgetter("query") | retriever, "query": itemgetter("query")}| rag_prompt | llm



class Input(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str = Field(desc="Input query to the model")

class Output(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    output: Any = Field(desc="Output of the model")


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# add_routes(
#     app, 
#     lcel_rag_chain.with_types(input_type=Input, output_type=Output).with_config(
#         {"run_name" : "EU"}
#     )
# )

add_routes(
    app, 
    lcel_rag_chain.with_types(input_type=Input, output_type=Output),
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
