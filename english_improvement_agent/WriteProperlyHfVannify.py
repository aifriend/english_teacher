from happytransformer import HappyTextToText, TTSettings

from english_improvement_agent.EnglishAgentInterface import EnglishAgentInterface


class WriteProperlyHfVannify(EnglishAgentInterface):

    def __init__(self):
        """
        vennify/t5-base-grammar-correction: This model generates a revised version of inputted text
        with the goal of containing fewer grammatical errors. It was trained with Happy Transformer
        using a dataset called JFLEG. Here's a full article on how to train a similar model.
        """
        super().__init__()
        self.args = TTSettings(num_beams=5, min_length=1)
        self.model = HappyTextToText(
            "T5", "vennify/t5-base-grammar-correction")

    def write_properly(self, input_text, max_length=256):
        input_text = f"grammar: {input_text}."
        result = self.model.generate_text(input_text, args=self.args)
        edited_text = result.text

        return edited_text
