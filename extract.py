'''import asyncio
import base64
import logging
from typing import Union, Dict
import aiohttp
from config import Settings

class DocumentIntelligenceService:
    """
    A service class for interacting with Azure Document Intelligence API.
    """

    def __init__(self):
        settings = Settings()
        self.key = settings.document_intelligence.api_key
        self.endpoint = settings.document_intelligence.endpoint
        self.api_version = "2024-11-30"

    async def analyze(
        self,
        source: Union[str, bytes],
        is_url: bool = True,
        model_id: str = "prebuilt-contract",
    ):
        """
        Analyze a document using Azure Document Intelligence.
        Yields each page as it arrives.
        """
        async with aiohttp.ClientSession() as session:
            # Submit analysis job, get operation result id
            result_id = await self._submit_analysis(session, source, model_id, is_url)

            # Poll and yield pages until analysis is done
            async for page in self._get_analysis_results(session, result_id, model_id):
                print("Got page:", page["pageNumber"])
                yield page  # You can yield to allow streaming consumption outside

    async def _submit_analysis(
        self,
        session: aiohttp.ClientSession,
        source: Union[str, bytes],
        model_id: str,
        is_url: bool = True
    ) -> str:
        """
        Submit a document for analysis to Azure Document Intelligence.
        Returns the operation ID.
        """
        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}:analyze?api-version={self.api_version}&outputContentFormat=markdown"
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.key,
        }

        if is_url:
            # If source is a URL, send it directly
            data = {"urlSource": source}
        else:
            # Base64 encoded content must be sent directly as base64Source
            if isinstance(source, bytes):
                pdf_base64 = base64.b64encode(source).decode('utf-8')
            elif isinstance(source, str):
                # Assume string is already base64 encoded
                pdf_base64 = source
            else:
                raise ValueError("source must be bytes or base64 string when is_url is False")

            data = {"base64Source": pdf_base64}

        async with session.post(url, headers=headers, json=data) as resp:
            resp.raise_for_status()
            op_location = resp.headers.get("operation-location") or resp.headers.get("Operation-Location")
            if not op_location:
                raise ValueError("Missing Operation-Location header in response")
            result_id = op_location.split("/")[-1].split('?')[0]
            return result_id

    async def _get_analysis_results(self, session: aiohttp.ClientSession, result_id: str, model_id: str):
        """
        Poll for analysis results and yield pages as they become available.
        """
        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}/analyzeResults/{result_id}?api-version={self.api_version}&outputContentFormat=markdown"
        headers = {"Ocp-Apim-Subscription-Key": self.key}
        processed_pages = set()

        while True:
            logging.info("Waiting for analysis to complete...")

            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()

            pages = data.get("analyzeResult", {}).get("pages", [])
            for page in pages:
                if page["pageNumber"] not in processed_pages:
                    yield page
                    processed_pages.add(page["pageNumber"])

            status = data.get("status", "").lower()
            if status in ["succeeded", "failed"]:
                return data

            await asyncio.sleep(2)

async def main():
    service = DocumentIntelligenceService()
    source_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    async for page in service.analyze(source=source_url, is_url=True):
        print(f"Processed page {page['pageNumber']}")
        print(page)

asyncio.run(main())'''

import logging
import requests
import time
from typing import Union, Dict
from config import get_settings
from typing_extensions import List


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

    def analyze(
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
        status, data = self._is_prev_analysis_present(model_id)

        if not status:
            result_id = self._submit_analysis(source, is_url, model_id)
            return self._get_analysis_results(result_id, model_id)
        else:
            return data

    def _submit_analysis(
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
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        operation_location = response.headers.get("Operation-Location")
        if not operation_location:
            raise ValueError("Operation-Location header is missing in the response.")

        return operation_location.split("/")[-1].split("?")[0]

    def _get_analysis_results(self, result_id: str, model_id: str) -> Dict:
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

        with open("result_id.txt", "w") as file:
            file.write(result_id)

        while True:
            logging.info("Waiting for analysis to complete.")
            time.sleep(2)
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data.get("status") in ["succeeded", "failed"]:
                return data
            
    # NOTE:- THIS FUNCTION CANNOT BE USED IN REAL APPLICATION ONLY FOR TESTING PURPOSE

    def _is_prev_analysis_present(self, model_id:str)-> Union[List,bool]:
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
        
        with open("result_id.txt", "r") as file:
            old_result_id = file.readline()

        if old_result_id is None:
            return [False, None]

        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}/analyzeResults/{old_result_id}?api-version={self.api_version}&outputContentFormat=markdown"
        headers = {"Ocp-Apim-Subscription-Key": self.key}

        while True:
            time.sleep(2)
            try:
                response = requests.get(url=url, headers=headers)
                response.raise_for_status()
                data = response.json()

                if data.get("status") in ["succeeded", "failed"]:
                    return [True,data]
            except Exception as e:
                print("Analysis deleted from server")
                return [False, None]


if __name__ == "__main__":
    # Example usage of the DocumentIntelligenceService
    pdf_blob_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    client = DocumentIntelligenceService()
    analysis_results = client.analyze(
        source=pdf_blob_url
    )
    print(analysis_results.keys())
    print(analysis_results["analyzeResult"].keys())
    print(analysis_results["analyzeResult"]["content"])
    print(analysis_results["analyzeResult"]["tables"])