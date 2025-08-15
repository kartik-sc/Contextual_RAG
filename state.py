from typing import List
from pydantic import BaseModel, Field

class ReasoningQueryPlan(BaseModel):
    key_concepts: List[str] = Field(description="The core terms related to the insurance, procedures, or concepts in the user's query.")
    reasoning_sub_questions: List[str] = Field(description="A list of implicit questions a claims adjuster would need to ask based on the key concepts, such as checking for exclusions, network status, or waiting periods.")
    

class QueryPlan(BaseModel):
    key_concepts: List[str] = Field(description="The core terms related to the insurance, procedures, or concepts in the user's query.")
    reasoning_sub_questions: List[str] = Field(description="A list of implicit questions a claims adjuster would need to ask based on the key concepts, such as checking for exclusions, network status, or waiting periods.")
    search_queries: List[str] = Field(description="A list of diverse, optimized search queries to send to the RAG system, derived from the sub-questions.")