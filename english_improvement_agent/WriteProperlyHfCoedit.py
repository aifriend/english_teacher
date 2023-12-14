from transformers import AutoTokenizer, T5ForConditionalGeneration

from english_improvement_agent.EnglishAgentInterface import EnglishAgentInterface


class WriteProperlyHfCoedit(EnglishAgentInterface):

    def __init__(self):
        """
        This model was obtained by fine-tuning the corresponding
        google/flan-t5-large model on the CoEdIT dataset. Details of the dataset
        can be found in our paper and repository. Given an edit instruction and
        an original text, our model can generate the edited version of the text.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
        self.model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")

    def write_properly(self, input_text, max_length=256):
        input_text = f'Fix grammatical errors in this sentence: {input_text}.'
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=max_length)
        edited_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return edited_text
