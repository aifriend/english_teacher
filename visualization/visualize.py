import json
import os

import gradio as gr
import requests
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


def put_one():
    global file_list
    global history_back
    file_list.append(history_back.pop())  # rollback last one
    f_path = history_back.pop()
    n_left = len(file_list)
    if n_left == 0:
        demo.close()
    return f_path, n_left


def get_one():
    global file_list
    global history_back
    f_path = file_list.pop()
    history_back.append(f_path)
    n_left = len(file_list)
    return f_path, n_left


def load_set(progress=gr.Progress()):
    global file_list
    global file_source
    file_list = list()
    for r, d, f in progress.tqdm(os.walk(file_source), desc="Loading doc"):
        for file in f:
            f_root, f_ext = os.path.splitext(file)
            if 'png'.lower() in f_ext.lower():
                file_list.append(os.path.join(r, file))

    return [0, len(file_list)]


def predict_mm():
    content_class = ''
    classes = {'Identity': .0, 'Pay bill': .0}
    return content_class, classes


def predict_llm(inputs):
    system_msg = f"""
        You are a helpful assistant who classifies any given customer text received by a banking company. 
        Please classify them into one of the given classes only, identity and pay bill. The identity class represents 
        various types of identification documents, such as identity cards, passports, driver's licenses that store 
        personal information and facilitate identification processes. The pay bill class represents a system for 
        handling bill payments that manage information related to bills, payments, and associated details.
    """

    # Queries the model with a given question and returns the answer.
    prompt = PromptTemplate(
        input_variables=["question_context"],
        template=("""
        Considering the provided text extracted from a document:

        {question_context}

        Can you determine the category to which this document likely belongs? Start your response with the name of the class chosen and then include an explanation for your choice.
        """),
    )
    prompt_question = prompt.format(
        question_context=inputs
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_gpt4_key}"
    }

    initial_message = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"{prompt_question}"},
        {"role": "assistant", "content": 'I guess this text belong to class'},
    ]
    payload = {
        "model": "gpt-4",
        "messages": initial_message,
        "temperature": 1.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    content_class = ''
    classes = {'Identity': .0, 'Pay bill': .0}
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload, stream=True)
    if response.status_code == 200:
        res = json.loads(response.text)
        content_class = res['choices'][0]['message']['content']
        if 'pay bill' in content_class or 'payslip' in content_class:
            classes = {'Identity': .0, 'Pay bill': 1.}
        elif 'identity' in content_class:
            classes = {'Identity': 1., 'Pay bill': .0}

    return content_class, classes


# to set a component as visible=False
def set_hidden():
    return gr.update(visible=False)


# to set a component as visible=True
def set_visible():
    return gr.update(visible=True)


with gr.Blocks() as demo:
    # Load the OpenAI API key from the environment variable
    load_dotenv()  # load the environment variables from the .env file
    openai_gpt4_key = os.getenv('OPENAI_API_KEY')
    class_id = 'class_0'
    root_path = r'C:\Users\JoseAntonioFernandez\Desktop\LLM\multimodal'
    with gr.Group():
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                show_doc = gr.Image(
                    label='Doc', type="pil", height=845, container=True)
            with gr.Column(scale=1):
                with gr.Row():
                    btn_next = gr.Button(
                        "Next >>", scale=1, variant='primary', visible=False, interactive=True)
                    btn_rem = gr.Button("Delete", scale=1, variant='stop')
                    btn_back = gr.Button("<< Back", scale=1, variant='secondary')
                    count_rn = gr.Number(label="Total removed", value=0)
                    count_tr = gr.Number(label="Total left", value=0)
                load_set_btn = gr.Button("Load Set")
                file_box = gr.Textbox(label="Doc")
                ocr_box = gr.Textbox(
                    lines=22, max_lines=22, label="OCR", autoscroll=False)
            with gr.Column(scale=1):
                llm_output = gr.Textbox(
                    lines=13, max_lines=13, label="LLM Response", autoscroll=False)
                class_llm_lab = gr.Label(
                    label="Class by LLM", num_top_classes=2, value={'Identity': .0, 'Pay bill': .0})
                llm_btn = gr.Button("Run LLM", variant='primary')
                class_mm_lab = gr.Label(
                    label="Class by MM", num_top_classes=2, value={'Identity': .0, 'Pay bill': .0})
                mm_btn = gr.Button("Run MM", variant='primary')

        load_set_btn.click(load_set,
                           None,
                           [count_rn, count_tr])
        load_set_btn.click(set_visible,
                           [],
                           [btn_next])

        llm_btn.click(predict_llm,
                      [ocr_box],
                      [llm_output, class_llm_lab])
        mm_btn.click(predict_mm,
                     None,
                     [llm_output, class_mm_lab])

if __name__ == "__main__":
    demo.launch(debug=True)
