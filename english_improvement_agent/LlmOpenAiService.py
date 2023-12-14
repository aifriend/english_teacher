import os

import openai
from dotenv import load_dotenv
from llama_index import download_loader
from llama_index.llms import OpenAI

from english_improvement_agent.LlmService import LlmService


class LlmOpenAiService(LlmService):
    """
    OpenAI LLM helper
    """

    def __init__(self, model_name=''):
        super().__init__()
        self.logger.Information("Init OpenAI")
        self.model_name = model_name
        load_dotenv()  # load the environment variables from the .env file
        self.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key
        self.embeddings = "text-embedding-ada-002"
        self.root_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../")

    def init(self, temp=1, max_token=256):
        self.llm = OpenAI(
            model=self.model_name,
            temperature=temp,
            max_tokens=max_token
        )

    def init_legacy(self):
        from openai import OpenAI
        self.llm = OpenAI()

    def load_web_docs(self, hint_list: list):
        wikipedia_reader = download_loader("WikipediaReader")
        loader = wikipedia_reader()
        self.documents = loader.load_data(pages=hint_list)

        return self.documents

    def get_doc_from_llama_index(self, ds_path):
        self.logger.Information(f'Get dataset from {ds_path}')
        simple_csv_reader = download_loader("PagedCSVReader")
        loader = simple_csv_reader()
        self.documents = loader.load_data(
            file=os.path.join(self.root_path, ds_path))

        return self.documents
