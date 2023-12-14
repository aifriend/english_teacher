from dotenv import load_dotenv
from huggingface_hub import login

from english_improvement_agent.LlmService import LlmService
from english_improvement_agent.commonsLib import loggerElk


class LlmHfService(LlmService):

    def __init__(self, model_name=''):
        super().__init__()
        self.model_name = model_name
        self.logger = loggerElk(__name__)

    def init(self):
        # Load the Huggingface API key from the environment variable
        load_dotenv()  # load the environment variables from the .env file
        login()
