from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from english_improvement_agent.EnglishAgentInterface import EnglishAgentInterface
from english_improvement_agent.LlmOpenAiService import LlmOpenAiService


class WriteProperlyInstruct(EnglishAgentInterface):

    def __init__(self):
        super().__init__()
        self.openai_serv = LlmOpenAiService(model_name="gpt-4")
        self.openai_serv.init_legacy()

        self.model = self.openai_serv

    def write_properly(self, input_text, max_length=256):
        response = self.model.llm.chat.completions.create(
            model=self.model.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Act as a english teacher. Your task is to focus solely on correcting grammatical errors in the provided text. Please identify and rectify any grammar-related issues, such as incorrect verb forms, subject-verb agreement problems, punctuation errors, and other syntactic mistakes. It is crucial to maintain the original meaning while ensuring that the revised text adheres to proper grammatical conventions. Avoid making stylistic changes; the primary goal is to enhance grammatical accuracy."
                },
                {
                    "role": "user",
                    "content": f"""
Translate this text into English that is grammatically correct, but do not change the style:

Original sentence:    
{input_text}

Desired output:
- Grammatically correct sentence
- Retain the original style of the sentence

Return only the corrected sentence.
                    """
                }
            ],
            temperature=0.5,
            max_tokens=2000,
            top_p=1
        )

        improved_text = response.choices[0].message.content

        return improved_text

    def write_the_same_grammar_fixed(self, input_text, max_length=4000):
        # load documents from the dataset JFLEG. JFLEG is for developing and
        # evaluating grammatical error correction (GEC).
        documents = self.model.get_doc_from_langchain(ds_path='data/jfleg.csv')

        self.logger.Information("Building vector store index")
        db = Chroma.from_documents(documents, OpenAIEmbeddings())
        retriever = db.as_retriever()
        retrieved_docs = retriever.invoke(input_text)
        context = retrieved_docs[0].page_content

        response = self.model.llm.chat.completions.create(
            model=self.model.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Your task is to refine and improve the grammar and style of the provided text. Ensure that the output is grammatically correct, coherent, and exhibits a sophisticated writing style. Pay attention to sentence structure, word choice, and overall fluency. While maintaining the original meaning, aim to elevate the expressiveness and clarity of the text. Feel free to rephrase, reword, and enhance the language as needed."
                },
                {
                    "role": "user",
                    "content": f"""
I'm working on improving my English writing skills. Can you help me rewrite this text to make it sound more natural and fluent, with better grammar and style?

Examples of sentence and corrections are:
{context}

Original sentence to be improved in grammar and style:
{input_text}

Desired corrections in output:
- Improved grammar and style
- Retain the original meaning of the sentence
- Generate a variety of outputs

Please rewrite the text for me, making the necessary corrections and improvements and generate multiples outputs.
                            """
                }
            ],
            temperature=0.5,
            max_tokens=2000,
            top_p=1
        )
        improved_text = response.choices[0].message.content

        return improved_text
