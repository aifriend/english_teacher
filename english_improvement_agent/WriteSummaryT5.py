from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from english_improvement_agent.EnglishAgentInterface import EnglishAgentInterface


class WriteSummaryT5(EnglishAgentInterface):

    def __init__(self):
        """
        With T5, we propose re-framing all NLP tasks into a unified text-to-text-format
        where the input and output are always text strings, in contrast to
        BERT-style models that can only output either a class label or a span
        of the input. Our text-to-text framework allows us to use the same model,
        loss function, and hyperparameters on any NLP task.
        """
        super().__init__()
        # Load the T5 tokenizer and model for summarization
        self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def summary(self, input_text, max_length=500):
        # Create the T5 prompt for summarization
        t5_prompt = f"Summarize this text: {input_text}"

        # Generate the summary using the T5 model
        t5_output = self.t5_model.generate(
            self.t5_tokenizer(t5_prompt, return_tensors="pt").input_ids,
            max_length=max_length,
            num_beams=4,
            no_repeat_ngram_size=2,
        )

        # Decode the generated summary
        summary = self.t5_tokenizer.batch_decode(
            t5_output, skip_special_tokens=True)[0]

        return summary
