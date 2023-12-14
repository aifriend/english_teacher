from english_improvement_agent.WriteProperlyInstruct import WriteProperlyInstruct
from english_improvement_agent.WriteStyleLlamaIndex import WriteStyleLlamaIndex
from english_improvement_agent.commonsLib import loggerElk


class EnglishImprovementAgentStyle:

    def __init__(self):
        self.logger = loggerElk(__name__)
        self.ll_idx_service = WriteStyleLlamaIndex()
        self.ins_service = WriteProperlyInstruct()

    def write_the_same_grammar_fixed_with_llama_index_rag(self, input_text):
        """
        Router query engine that selects one out of several candidate query engines to execute a query.
        Use two type of tools one for vector search on JFLEG dataset and the other to summarize content as needed.

        :param input_text:
        :return:
        """
        self.logger.Information("Write good style with RAG and llama_index")
        response = self.ll_idx_service.write_the_same_grammar_fixed(input_text)
        return response

    def write_the_same_grammar_fixed_with_llm_instruction(self, input_text):
        """
        simple RAG architecture with Q&A downstream task and
        vector search in JFLEG dataset and effective prompting engineering

        :param input_text:
        :return:
        """
        self.logger.Information("Write good style with RAG and Q&A plus vector search retrieval")
        response = self.ins_service.write_the_same_grammar_fixed(input_text)
        return response
