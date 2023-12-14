from langchain_core.prompts import PromptTemplate
from llama_index import ServiceContext, VectorStoreIndex, SummaryIndex, PromptHelper, OpenAIEmbedding, Prompt, \
    get_response_synthesizer
from llama_index.logger import LlamaLogger
from llama_index.prompts import PromptType
from llama_index.query_engine import RouterQueryEngine
from llama_index.response_synthesizers import ResponseMode
from llama_index.tools import QueryEngineTool, ToolMetadata

from english_improvement_agent.EnglishAgentInterface import EnglishAgentInterface


class WriteStyleLlamaIndex(EnglishAgentInterface):

    def __init__(self):
        super().__init__()
        self.logger.Information(
            "Load llama-index RAG chain for Openai GPT4 model")
        self.openai_serv.init(temp=0.5, max_token=2000)

        # create the service context
        self.logger.Information("Loading RAG docs from JFLEG database")
        prompt_helper = PromptHelper(
            context_window=4096,
            num_output=256,
            chunk_overlap_ratio=0.1,
            chunk_size_limit=None
        )
        self.service_context = ServiceContext.from_defaults(
            llm=self.openai_serv.llm,
            embed_model=OpenAIEmbedding(),
            prompt_helper=prompt_helper,
            llama_logger=LlamaLogger()
        )

        # load documents from the dataset JFLEG
        documents = self.openai_serv.get_doc_from_llama_index(ds_path='data/jfleg.csv')

        # create vector index and summary_index tools
        self.logger.Information("Building vector store index")
        self.vector_index = VectorStoreIndex.from_documents(
            documents, service_context=self.service_context)
        # TODO: improve performance loading storage_context if persisted
        # vector_store_path = os.path.join(self.openai_serv.root_path, 'data/index')
        # self.vector_index.storage_context.persist(
        #     persist_dir=vector_store_path)
        self.logger.Information("Building vector store summary index")
        self.summary_index = SummaryIndex.from_documents(
            documents, service_context=self.service_context)

        # Query engine tool to search for related sentences and correction
        # to be used as a few-shots template for an in-context learning effective prompting approach
        vector_tool = QueryEngineTool(
            self.vector_index.as_query_engine(),
            metadata=ToolMetadata(
                name="vector_search",
                description="Useful for searching for related or similar examples of sentences with corrections."
            )
        )

        # Summary engine tool to summarize an entire sentence to be used as
        # an in-context learning effective prompting approach throwout
        # Q&A and Refinement router query agent
        summary_tool = QueryEngineTool(
            self.summary_index.as_query_engine(response_mode="tree_summarize"),
            metadata=ToolMetadata(
                name="summary",
                description="Useful for summarizing an entire sentence."
            )
        )

        self.query_engine = RouterQueryEngine.from_defaults(
            [vector_tool, summary_tool],
            service_context=self.service_context,
            select_multi=False
        )

    def write_the_same_grammar_fixed(self, input_text, max_length=4000):
        rag_prompt = PromptTemplate(
            input_variables=["input_text"],
            template=("""
Your task is to refine and improve the grammar and style of the provided text. Ensure that the output is grammatically correct, coherent, and exhibits a sophisticated writing style. Pay attention to sentence structure, word choice, and overall fluency. While maintaining the original meaning, aim to elevate the expressiveness and clarity of the text. Feel free to rephrase, reword, and enhance the language as needed.

I'm working on improving my English writing skills. Can you help me rewrite this text to make it sound more natural and fluent, with better grammar and style?

Original sentence to be improved in grammar and style:
{input_text}

Desired corrections in output:
- Improved grammar and style
- Retain the original meaning of the sentence
- Generate a variety of outputs

Please rewrite the text for me, making the necessary corrections and improvements and generate multiples outputs\n
"""
                      ),
        )
        # The augmented context for the input to handle ambiguities and context-specific nuances in user inputs
        rag_question = rag_prompt.format(
            input_text=input_text
        )
        improved_text = self.query_engine.query(rag_question)

        return improved_text

    @staticmethod
    def default_template():
        """
        vector_index.as_query_engine(response_synthesizer=response_synthesizer)

        :return:
        """
        DEFAULT_TEXT_QA_PROMPT_TMPL = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the question: {query_str}\n"
        )
        DEFAULT_REFINE_PROMPT_TMPL = (
            "The original question is as follows: {query_str}\n"
            "We have provided an existing answer: {existing_answer}\n"
            "We have the opportunity to refine the existing answer "
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the question. "
            "If the context isn't useful, return the original answer."
        )
        DEFAULT_SIMPLE_INPUT_TMPL = "{query_str}"
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            text_qa_template=Prompt(DEFAULT_TEXT_QA_PROMPT_TMPL,
                                    prompt_type=PromptType.QUESTION_ANSWER),
            refine_template=Prompt(DEFAULT_REFINE_PROMPT_TMPL,
                                   prompt_type=PromptType.REFINE),
            simple_template=Prompt(DEFAULT_SIMPLE_INPUT_TMPL,
                                   prompt_type=PromptType.SIMPLE_INPUT)
        )

        return response_synthesizer
