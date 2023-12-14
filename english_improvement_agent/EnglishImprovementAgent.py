from dotenv import load_dotenv

from english_improvement_agent.EnglishImprovementAgentStyle import EnglishImprovementAgentStyle
from english_improvement_agent.EnglishImprovementAgentSummary import EnglishImprovementAgentSummary
from english_improvement_agent.EnglishImprovementAgentWrite import EnglishImprovementAgentWrite
from english_improvement_agent.commonsLib import loggerElk


class EnglishImprovementAgent:

    def __init__(self):
        self.logger = loggerElk(__name__)
        load_dotenv()  # load the environment variables from the .env file
        # factory of agents for writing properly, writing with style and summarization
        self.write_service = EnglishImprovementAgentWrite()
        self.style_service = EnglishImprovementAgentStyle()
        self.summary_service = EnglishImprovementAgentSummary()

    def write_properly(self, input_text):
        # input_text = "This sentences has has bads grammar"
        # input_text = "When I grow up, I start to understand what he said is quite right"
        return self.write_service.write_properly_vannity(input_text)

    def write_the_same_grammar_fixed(self, input_text):
        # input_text = "It's fantastic to have a doctor like you—so talented, clear-minded, and focused on your expertise. I tried a local doctor, and it was disappointing. You're undoubtedly one of the best in treating floaters. I hope to see you soon as your patient."
        return self.style_service.write_the_same_grammar_fixed_with_llama_index_rag(input_text)
        # return self.style_service.write_the_same_grammar_fixed_with_llm_instruction(input_text)

    def summarize(self, input_text):
        # input_text = "The Speakeasy series’ first video is illustrating a very simple truth: if someone cannot communicate effectively, then the person won’t be able to present their own ideas or knowledge to an audience."
        return self.summary_service.summarization_with_t5(input_text)
