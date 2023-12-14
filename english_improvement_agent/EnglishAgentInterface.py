from dotenv import load_dotenv

from english_improvement_agent.commonsLib import loggerElk


class EnglishAgentInterface:

    def __init__(self):
        self.logger = loggerElk(__name__)
        load_dotenv()  # load the environment variables from the .env file
        self.tokenizer = None
        self.model = None

    def write_properly(self, input_text, max_length=256):
        """
        Enhances both grammar and style of the input message.

        Args:
          input_text (str): The input text to be improved.
          max_length (int): Max length of th output text.

        Returns:
          str: The improved text.
        """
        pass

    def write_the_same_grammar_fixed(self, input_text, max_length=256):
        """
        Corrects only the grammatical errors in the input message.

        Args:
          input_text (str): The input text to be improved.
          max_length (int): Max length of th output text.

        Returns:
          str: The improved text.
        """
        pass

    def summary(self, input_text, max_length=256):
        """
        Provides a concise summary of the input message.

        Args:
          input_text (str): The input text to be summarized.
          max_length (int): Max length of th output text.

        Returns:
          str: The summary of the input text.
        """
        pass
