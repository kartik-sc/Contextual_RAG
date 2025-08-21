# [NOT NECESSARY] Add an extra node in the beginnig where the user himself can add a PDF if needed

import uuid
from langgraph.graph import StateGraph
from langgraph.graph import END, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, StateSnapshot
from IPython.display import Image, display, Markdown
from typing import TypedDict
from dotenv import load_dotenv
import asyncio

from chunker import TableAwareChunker
from contextualRAG import ContextualizedRAG
from state import InputState, OutputState, OverallState

load_dotenv()
class UserInteraction:
    def __init__(self):
        self.crag = ContextualizedRAG()
        self.pdf_blob_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        self.graph = None
        self.data_loaded = False
    
    def build_graph(self):
        """
        Builds a graph workflow for the system
        """

        builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
        checkpointer = InMemorySaver()

        builder.add_node("Parse_and_reason_query", self.crag.get_parse_user_query)
        builder.add_node("Ask_reason_response", self.crag._human_in_loop)
        builder.add_node("Generate_search_queries", self.crag._generate_search_queries)
        builder.add_node("Retrieve_content", self.crag._retrieve_and_grade)
        builder.add_node("Generate_answer", self.crag._generate_answer)
        builder.add_node("wait_for_data", self.crag._wait_for_data)
        builder.add_node("fallback", self.crag.fallback_node)

        builder.add_edge(START,"Parse_and_reason_query")
        builder.add_edge("Parse_and_reason_query","Ask_reason_response")
        builder.add_edge("Ask_reason_response","Generate_search_queries")
        builder.add_edge("Generate_search_queries", "Retrieve_content")
        builder.add_edge("Retrieve_content","Generate_answer")
        builder.add_edge("Generate_answer", END)

        self.graph = builder.compile(checkpointer=checkpointer)

    async def interact(self, ingestion_task:asyncio.create_task, pdf_source:str = None, user_query:str = None):
        """
        Synchronous function part which deals with the user interaction
        """
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        if pdf_source:
            self.pdf_blob_url = pdf_source

        result = await self.graph.ainvoke(
            InputState(user_query=user_query, pdf_blob_url=self.pdf_blob_url), 
            config=config
        )  

        while "__interrupt__" in result:
            print(f"Question: {result['__interrupt__']}")
            answer = await asyncio.to_thread(input, "Your answer: ")
            result = await self.graph.ainvoke(Command(resume=answer), config=config)

        print("\nWaiting for data ingestion to complete.")
        
        ingestion_result = await ingestion_task

        if ingestion_result["error"][0]:
            error = ingestion_result['error'][1]
            print(f"Data ingestion failed: {error}")
            await self.graph.aupdate_state(config, {"error": [True, error]})
        else:
            print("Data ingestion completed \n")
            await self.graph.aupdate_state(config, {"data_loaded": True, "error": [False, ""]})

        final_result = await self.graph.ainvoke(None, config=config)
        print(final_result["final_response"])

    async def wrapper(self):
        """
        Main entrypoint to run the system.
        """

        self.build_graph()
        query = "undergone a treatment for thyroid. What is the coverage in my policy"

        print("Starting data ingestion in the background...")
        ingestion_state = InputState(user_query=query, pdf_blob_url=self.pdf_blob_url)
        
        ingestion_task = asyncio.create_task(self.crag.data_ingestion(ingestion_state))

        await self.interact(ingestion_task, user_query=query)


async def main():
    ui = UserInteraction()
    await ui.wrapper()

if __name__ == "__main__":
    asyncio.run(main())

