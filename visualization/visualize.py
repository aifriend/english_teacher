import gradio as gr
from dotenv import load_dotenv

from english_improvement_agent.EnglishImprovementAgent import EnglishImprovementAgent

english_teacher = EnglishImprovementAgent()


def predict_llm1(inputs, input_exp, drops):
    request = inputs if inputs else (input_exp if input_exp else '')
    if drops == 'vannity':
        response = english_teacher.write_properly_0(request)
    elif drops == 'coedit':
        response = english_teacher.write_properly_1(request)
    else:
        response = english_teacher.write_properly_2(request)
    return '', request, response


def predict_llm2(inputs, input_exp, drops):
    request = inputs if inputs else (input_exp if input_exp else '')
    if drops == 'llama_index':
        response = english_teacher.write_the_same_grammar_fixed_0(request)
    else:
        response = english_teacher.write_the_same_grammar_fixed_1(request)
    return '', request, response


def predict_llm3(inputs, input_exp):
    request = inputs if inputs else (input_exp if input_exp else '')
    response = english_teacher.summarize(request)
    return '', request, response


with gr.Blocks() as teacher:
    load_dotenv()
    with gr.Group():
        ocr_box = gr.Textbox(
            lines=3, max_lines=3, label="New Request", autoscroll=False)
        with gr.Tab("Write properly"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    ll_drop1 = gr.Dropdown(["vannity", "coedit", "openai"], label="Model selection")
                    llm_input1 = gr.Textbox(
                        value="It fantast to have a doctor like you—so talented, clear-minded, and focused on yours expertise. I try a local doctor, and it was disappointing. ",
                        lines=3, max_lines=3, label="LLM Request", autoscroll=False)
                    llm_output1 = gr.Textbox(
                        lines=3, max_lines=3, label="LLM Response", autoscroll=False, interactive=False)
                    llm_btn1 = gr.Button(
                        "Enhances both grammar and style of the input message", variant='primary')
        with gr.Tab("Write the same grammar fixed"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    ll_drop2 = gr.Dropdown(["llama_index", "instruct"], label="Model selection")
                    llm_input2 = gr.Textbox(
                        value="It's fantastic to have a doctor like you—so talented, clear-minded, and focused on your expertise. I tried a local doctor, and it was disappointing. You're undoubtedly one of the best in treating floaters. I hope to see you soon as your patient.",
                        lines=3, max_lines=3, label="LLM Request", autoscroll=False)
                    llm_output2 = gr.Textbox(
                        lines=15, max_lines=15, label="LLM Response", autoscroll=True, interactive=False)
                    llm_btn2 = gr.Button(
                        "Corrects only the grammatical errors in the input message", variant='primary')
        with gr.Tab("Summarize"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    llm_input3 = gr.Textbox(
                        value="It's fantastic to have a doctor like you—so talented, clear-minded, and focused on your expertise. I tried a local doctor, and it was disappointing. You're undoubtedly one of the best in treating floaters. I hope to see you soon as your patient.",
                        lines=3, max_lines=3, label="LLM Request", autoscroll=False)
                    llm_output3 = gr.Textbox(
                        lines=3, max_lines=3, label="LLM Response", autoscroll=False, interactive=False)
                    llm_btn3 = gr.Button(
                        "Provides a concise summary of the input message", variant='primary')

        llm_btn1.click(predict_llm1,
                       [ocr_box, llm_input1, ll_drop1],
                       [ocr_box, llm_input1, llm_output1])
        llm_btn2.click(predict_llm2,
                       [ocr_box, llm_input2, ll_drop2],
                       [ocr_box, llm_input2, llm_output2])
        llm_btn3.click(predict_llm3,
                       [ocr_box, llm_input3],
                       [ocr_box, llm_input3, llm_output3])

if __name__ == "__main__":
    teacher.launch(debug=True)
