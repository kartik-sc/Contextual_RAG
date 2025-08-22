import logging
import requests
import time
from typing import Union, Dict
from config.config import get_settings
from typing_extensions import List
import asyncio
import aiohttp # Import aiohttp

class DocumentIntelligenceService:
    """
    A service class for interacting with Azure Document Intelligence API.
    This class provides methods to analyze documents using Azure's Document Intelligence service.
    """

    def __init__(self):
        """
        Initialize the DocumentIntelligenceService with API credentials and endpoint.
        """
        settings = get_settings()
        self.key = settings.document_intelligence.api_key
        self.endpoint = settings.document_intelligence.endpoint
        self.api_version = "2024-11-30" 

    async def analyze(
        self,
        source: Union[str, bytes],
        is_url: bool = True,
        model_id: str = "prebuilt-layout",
    ) -> Dict:
        """
        Analyze a document using Azure Document Intelligence.
        Args:
            source (Union[str, bytes]): The document source, either a URL or base64 encoded content.
            is_url (bool): True if the source is a URL, False if it's base64 encoded content.
            model_id (str): The ID of the model to use for analysis.
        Returns:
            Dict: The analysis results.
        Raises:
            requests.HTTPError: If the API request fails.
        """
        status, data = await self._is_prev_analysis_present(model_id)

        if not status:
            result_id = await self._submit_analysis(source, is_url, model_id)
            result = await self._get_analysis_results(result_id, model_id)
            return result
        else:
            return data

    async def _submit_analysis(
        self, source: Union[str, bytes], is_url: bool, model_id: str
    ) -> str:
        """
        Submit a document for analysis to Azure Document Intelligence.
        Args:
            source (Union[str, bytes]): The document source, either a URL or base64 encoded content.
            is_url (bool): True if the source is a URL, False if it's base64 encoded content.
            model_id (str): The ID of the model to use for analysis.
        Returns:
            str: The result ID for the submitted analysis.
        Raises:
            ValueError: If the Operation-Location header is missing in the response.
            requests.HTTPError: If the API request fails.
        """
        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}:analyze?api-version={self.api_version}&outputContentFormat=markdown"
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.key,
        }
        data = {"urlSource": source} if is_url else {"base64Source": source}

        logging.info("Submitting document for analysis")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                operation_location = response.headers.get("Operation-Location")
                if not operation_location:
                    raise ValueError("Operation-Location header is missing in the response.")
                return operation_location.split("/")[-1].split("?")[0]

    async def _get_analysis_results(self, result_id: str, model_id: str) -> Dict:
        """
        Retrieve the analysis results from Azure Document Intelligence.
        Args:
            result_id (str): The ID of the analysis result to retrieve.
            model_id (str): The ID of the model used for analysis.
        Returns:
            Dict: The analysis results.
        Raises:
            requests.HTTPError: If the API request fails.
        """
        
        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}/analyzeResults/{result_id}?api-version={self.api_version}&outputContentFormat=markdown"
        headers = {"Ocp-Apim-Subscription-Key": self.key}

        with open("config/result_id.txt", "w") as file:
            file.write(result_id)

        async with aiohttp.ClientSession() as session:
            while True:
                logging.info("Waiting for analysis to complete.")
                await asyncio.sleep(2)
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if data.get("status") in ["succeeded", "failed"]:
                        return data
            
    # NOTE:- THIS FUNCTION CANNOT BE USED IN REAL APPLICATION ONLY FOR TESTING PURPOSE

    async def _is_prev_analysis_present(self, model_id:str)-> Union[List,bool]:
        """
        Checks if the analysis of that document is already present in the azure document store
        Args:
            model_id (str): The ID of the model used for analysis.
        Returns:
            List: The status and analysis results.
        Raises:
            requests.HTTPError: If the API request fails.

        NOTE:-  THIS FUNCTION CANNOT BE USED IN REAL APPLICATION ONLY FOR TESTING PURPOSE 
                APPLICABLE ONLY IF THE SOURCE URL IS THE SAME
        """
        
        try:
            with open("config/result_id.txt", "r") as file:
                old_result_id = file.readline().strip()
        except FileNotFoundError:
            return [False, None]

        if not old_result_id:
            return [False, None]

        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}/analyzeResults/{old_result_id}?api-version={self.api_version}&outputContentFormat=markdown"
        headers = {"Ocp-Apim-Subscription-Key": self.key}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url=url, headers=headers) as response:
                    if response.status == 404:
                        print("Analysis not found on server.")
                        return [False, None]
                    response.raise_for_status()
                    data = await response.json()

                    if data.get("status") == "succeeded":
                        return [True, data]
                    else:
                        # If status is running or other, treat as not present for simplicity
                        return [False, None]
            except aiohttp.ClientError as e:
                print(f"Analysis deleted from server or network error: {e}")
                return [False, None]


if __name__ == "__main__":
    # Example usage of the DocumentIntelligenceService
    pdf_blob_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    async def run_analysis():
        client = DocumentIntelligenceService()
        analysis_results = await client.analyze(source=pdf_blob_url)
        print(analysis_results.keys())
        print(analysis_results["analyzeResult"].keys())
        print(analysis_results["analyzeResult"]["content"])
        print(analysis_results["analyzeResult"]["tables"])

    asyncio.run(run_analysis())