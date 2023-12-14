from english_improvement_agent.WriteProperlyHfCoedit import WriteProperlyHfCoedit
from english_improvement_agent.WriteProperlyHfVannify import WriteProperlyHfVannify
from english_improvement_agent.WriteProperlyInstruct import WriteProperlyInstruct
from english_improvement_agent.commonsLib import loggerElk


class EnglishImprovementAgentWrite:

    def __init__(self):
        self.logger = loggerElk(__name__)
        self.van_service = WriteProperlyHfVannify()
        self.coe_service = WriteProperlyHfCoedit()
        self.ins_service = WriteProperlyInstruct()

    def write_properly_vannity(self, input_text):
        """
        This model generates a revised version of inputted text
        with the goal of containing fewer grammatical errors

        :param input_text:
        :return:
        """
        self.logger.Information("Write properly with HF Vannity")
        response = self.van_service.write_properly(input_text)
        return response

    def write_properly_coedit(self, input_text):
        """
        This model was obtained by fine-tuning the corresponding
        google/flan-t5-large model on the CoEdIT dataset

        :param input_text:
        :return:
        """
        self.logger.Information("Write properly with HF Coedit")
        response = self.coe_service.write_properly(input_text)
        return response

    def write_properly_instruct(self, input_text):
        """
        OpenAI gpt-4 model with in context learning and prompting engineering

        :param input_text:
        :return:
        """
        self.logger.Information("Write properly with OpenaAI Instruct")
        response = self.ins_service.write_properly(input_text)
        return response
