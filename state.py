from typing import List, Union, TypedDict
from pydantic import BaseModel, Field

class InputState(BaseModel):
    user_query: str = Field(description="Accepts the input query from user", default="")
    pdf_blob_url: str = Field(default="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D")

class ReasoningQueryPlan(BaseModel):
    user_query: str = Field(description="Accepts the input query from user")
    key_concepts: List[str] = Field(description="The core terms related to the insurance, procedures, or concepts in the user's query.")
    reasoning_sub_questions: List[str] = Field(description="A list of implicit questions a claims adjuster would need to ask based on the key concepts, such as checking for exclusions, network status, or waiting periods.")

class SearchQueryPlan(BaseModel):
    user_query: str = Field(description="Accepts the input query from user")
    key_concepts: List[str] = Field(description="The core terms related to the insurance, procedures, or concepts in the user's query.", default=[""])
    reasoning_sub_questions: List[str] = Field(description="These Queries are already generated just return the queries here.", default=[""])
    reasoning_responses: List[str] = Field(description="These responses are already provided by the user, just return the responses here.", default=[""])
    search_queries: List[str] = Field(description="A list of diverse, optimized search queries to send to the RAG system for good retrieval using the provided information by the user.", default=[""])

class OverallState(TypedDict):
    user_query: str = Field(description="Accepts the input query from user")
    key_concepts: List[str] = Field(description="The core terms related to the insurance, procedures, or concepts in the user's query.", default=[""])
    reasoning_sub_questions: List[str] = Field(description="These Queries are already generated just return the queries here.", default=[""])
    reasoning_responses: List[str] = Field(description="These responses are already provided by the user, just return the responses here.", default=[""])
    search_queries: List[str] = Field(description="A list of diverse, optimized search queries to send to the RAG system for good retrieval using the provided information by the user.", default=[""])
    top_chunks: List[List[Union[float, str]]] = Field(description="Stores top_n relevant chunks",default=[""])
    final_response: str = Field(description="Returns the final answer", default="")

class OutputState(BaseModel):
    final_response: str = Field(description="Returns the final answer")