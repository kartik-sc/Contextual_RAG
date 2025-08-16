# [NOT NECESSARY] Add an extra node in the beginnig where the user himself can add a PDF if needed

import uuid
from langgraph.graph import StateGraph
from langgraph.graph import END, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from IPython.display import Image, display
from typing import TypedDict

from chunker import TableAwareChunker
from contextualRAG import ContextualizedRAG
from state import InputState, OutputState, OverallState

crag = ContextualizedRAG()
pdf_blob_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
checkpointer = InMemorySaver()

builder.add_node("Load_data",crag.data_ingestion)
builder.add_node("Parse_and_reason_query", crag.get_parse_user_query)
builder.add_node("Ask_reason_response", crag._human_in_loop)
builder.add_node("Generate_search_queries", crag._generate_search_queries)
builder.add_node("Retrieve_content", crag._retrieve_and_grade)
builder.add_node("Generate_answer", crag._generate_answer)

builder.add_edge(START,"Load_data")
builder.add_edge("Load_data","Parse_and_reason_query")
builder.add_edge("Parse_and_reason_query","Ask_reason_response")
builder.add_edge("Ask_reason_response","Generate_search_queries")
builder.add_edge("Generate_search_queries", "Retrieve_content")
builder.add_edge("Retrieve_content","Generate_answer")
builder.add_edge("Generate_answer", END)

graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

result = graph.invoke(InputState(user_query="I have undergone a root canal dental treatment. What is the coverage in my policy"), config=config)
print(result["__interrupt__"])  

while True:
    answer = input("")
    result = graph.invoke(Command(resume=answer), config=config)
    if "__interrupt__" in result:
        print(result["__interrupt__"])
    else:
        break

print(result["final_response"])