from english_improvement_agent.WriteSummaryT5 import WriteSummaryT5
from english_improvement_agent.commonsLib import loggerElk


class EnglishImprovementAgentSummary:

    def __init__(self):
        self.logger = loggerElk(__name__)
        self.t5_service = WriteSummaryT5()

    def summarization_with_t5(self, input_text):
        """
        summarize input text based on T5 encoder-decoder model and converts all NLP problems into a text-to-text format

        :param input_text:
        :return:
        """
        self.logger.Information("Summarize with HF T5")
        response = self.t5_service.summary(input_text)
        return response
