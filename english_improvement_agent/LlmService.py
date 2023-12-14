import json

from english_improvement_agent.commonsLib import loggerElk


class LlmService:

    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.documents = list()
        self.doc_splits = list()
        self.logger = loggerElk(__name__)

    def init(self):
        """
        Initialize LangChain Models

        You initialize LangChain Models with the pre-trained text, chat and embeddings generation LLM model
        """

    @staticmethod
    def load_web_docs(source_path):
        return list()

    def get_doc_from_llama_index(self, ds_path):
        self._formatter_simple(response='')
        return list()

    def get_doc_from_langchain(self, ds_path):
        self._formatter_simple(response='')
        return list()

    def _formatter_simple(self, response, query=''):
        self.logger.Information("=" * 80)
        if query:
            print(f"Query: {query}")
        if isinstance(response, dict):
            self.logger.Information(f"Response: {json.dumps(response, indent=2)}")
        else:
            self.logger.Information(f"Response: {response}")
        self.logger.Information("=" * 80)
