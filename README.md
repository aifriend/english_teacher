# English Improvement Agent

This Python-based application leverages the OpenAI API and Transformers library to assist non-native English speakers in improving their written English. It provides three primary functionalities:

- **write_properly**: Enhances both grammar and style of the input message.
- **write_the_same_grammar_fixed**: Corrects only the grammatical errors in the input message.
- **summarization_with_t5**: Provides a concise summary of the input message.

## Setup and Usage

To set up and run the application, follow these steps:

1. Clone the repository.
2. Install the required Python packages using `pip install -r requirements.txt`.
3. Set your OpenAI API key in the `config.py` file.
4. Run the application using `python main.py`.

## Prompt Engineering for RAG

To differentiate between the 'write_properly' and 'write_the_same_grammar_fixed' functions, we use the following prompts:

- **write_properly**: "Rewrite the following text to improve both grammar and style, making it sound more natural and fluent in English. Ensure that the meaning and context of the original message are preserved."
- **write_the_same_grammar_fixed**: "Correct any grammatical errors in the following text without changing the overall meaning or style of the message."

We consider the following factors to ensure the model understands the distinction between style enhancement and mere grammatical correction:

- **Use clear and concise instructions**: The prompts are written in a straightforward manner, avoiding ambiguous or complex language.
- **Emphasize the desired outcome**: The prompts explicitly state the goal of each function, whether it's improving grammar, style, or both.
- **Provide examples**: The prompts include examples of input and output to help the model understand the expected behavior.

## API Utilization Strategy

We utilize the OpenAI API efficiently and effectively by:

- **Batching requests**: We send multiple requests to the API in a single batch, reducing the number of API calls and optimizing performance.
- **Caching responses**: We cache the API responses to avoid sending duplicate requests for the same input.
- **Handling rate limits**: We implement a rate limiting mechanism to ensure we don't exceed the API's request limits.

## Handling Ambiguity in User Inputs

To handle ambiguities in user inputs, we employ the following strategies:

- **Contextual understanding**: We use the context of the input message to disambiguate ambiguous words or phrases.
- **User interaction**: We provide the user with the option to clarify ambiguous parts of their input, improving the accuracy of the output.
- **Fallback mechanism**: In cases where ambiguity cannot be resolved, we provide a fallback response that acknowledges the ambiguity and suggests alternative interpretations.

## Summarization Technique

We implement the 'Summarize' function using the following approach:

- **Text summarization model**: We use a pre-trained text summarization model from the Transformers library.
- **Input preprocessing**: We preprocess the input message to remove unnecessary details and extract the main points.
- **Summary generation**: We generate a summary of the input message using the summarization model.
- **Post-processing**: We post-process the summary to ensure it is coherent, concise, and retains the essential information from the original message.

## Performance Metrics and Evaluation

We use the following metrics to evaluate the performance of each function in the application:

- **Accuracy**: We measure the accuracy of the 'write_properly' and 'write_the_same_grammar_fixed' functions by comparing the output with human-generated corrections.
- **Fluency**: We evaluate the fluency of the output by measuring its readability, naturalness, and coherence.
- **Conciseness**: We assess the conciseness of the 'summarization_with_t5' function by measuring the length of the summary relative to the original message.
- **Informativeness**: We evaluate the informativeness of the summary by measuring its ability to capture the main points of the original message.

We gather feedback or data to improve the system continuously through user surveys, error analysis, and A/B testing. Prompt engineering plays a crucial role in this process by allowing us to fine-tune the prompts to achieve better performance and handle edge cases more effectively.

## Technical Questions

### Prompt Engineering for RAG

To differentiate between the 'write_properly' and 'write_the_same_grammar_fixed' functions in the RAG model prompts, we can use the following strategies:

1. **Explicit Instructions:** 
* In the prompt for 'write_properly', we can explicitly instruct the model to enhance both grammar and style. For example: "Your task is to refine and improve the grammar and style of the provided text. Ensure that the output is grammatically correct, coherent, and exhibits a sophisticated writing style. Pay attention to sentence structure, word choice, and overall fluency. While maintaining the original meaning, aim to elevate the expressiveness and clarity of the text. Feel free to rephrase, reword, and enhance the language as needed.". 
* In the prompt for 'write_the_same_grammar_fixed', we can give the same detailed instruction to the model only for grammatical error corrections. For example: "Act as a english teacher. Your task is to focus solely on correcting grammatical errors in the provided text. Please identify and rectify any grammar-related issues, such as incorrect verb forms, subject-verb agreement problems, punctuation errors, and other syntactic mistakes. It is crucial to maintain the original meaning while ensuring that the revised text adheres to proper grammatical conventions. Avoid making stylistic changes; the primary goal is to enhance grammatical accuracy."

2. **Contrastive Examples:** We can provide contrasting examples to help the model understand the difference between style enhancement and grammatical correction. 
* Style Enhancement Example: In this example, the stylistic enhancement involves replacing common words with more sophisticated alternatives, creating a more refined and descriptive expression while preserving the overall meaning.
```
    Original Text:
    "The old house was really big, and it had a large, expansive garden."

    Style-Enhanced Version:
    "The ancient dwelling boasted considerable proportions and featured an extensive garden."
```
* Style Enhancement Example: This style-enhanced version uses more descriptive language to convey the team's performance, enhancing the overall style of expression.
```
    Original Text:
    "The team played really well."

    Style-Enhanced Version:
    "The team exhibited exceptional prowess on the field."
```
* Grammatical Correction Example: this case, the focus is on fixing grammatical errors, such as the use of "don't" instead of "doesn't" and "eat" instead of "eats." The goal is to improve the accuracy and adherence to proper grammar rules.
```
    Original Text:
    "She don't like pizza, but she eat tacos all the time."

    Grammatically Corrected Version:
    "She doesn't like pizza, but she eats tacos all the time."
```
* Grammatical Correction Example: Here, the correction involves changing "was" to "were" to ensure subject-verb agreement, addressing a grammatical error.
```
    Original Text:
    "They was happy to receive the award."

    Grammatically Corrected Version:
    "They were happy to receive the award."
```

### API Utilization Strategy

To ensure efficient and effective use of the OpenAI API, we can consider the following strategies:

1. **Batch Processing:** Instead of sending each user input individually to the API, we can batch multiple inputs together and send them as a single request. This can significantly reduce the number of API calls and improve efficiency.

2. **Caching:** We can cache the responses from the API to avoid sending redundant requests for the same input. This can be especially useful for common phrases or sentences that users might input frequently.

3. **Rate Limiting:** We should implement rate limiting to avoid exceeding the API's request limits. This can be done by setting a maximum number of requests per unit time or using a queuing system to manage requests.

### Handling Ambiguity in User Inputs

To handle ambiguities and context-specific nuances in user inputs, we can employ the following techniques:

1. **Contextual Prompts:** We can provide additional context to the RAG model by including information about the user's intent or the topic of the input text. This can help the model better understand the meaning and nuances of the input.

2. **Multiple Outputs:** We can generate multiple outputs for each input and allow the user to select the most appropriate one. This gives the user more control over the output and allows them to choose the interpretation that best fits their intended meaning.

3. **User Feedback:** We can collect feedback from users on the generated outputs and use it to improve the prompts and the model's performance over time.

### Summarization Technique

To implement the 'Summarize' function, we can use the following approach:

1. **RAG-based Retrieval:** We can use the RAG model to retrieve relevant documents or passages from a large corpus that are related to the input text. This can help us gather important information and context for the summarization task.

2. **T5-based Generation:** We can then use the T5 model to generate a summary based on the retrieved documents and the input text. The T5 model can be fine-tuned on a summarization dataset to improve its performance.

3. **Length Control:** We can control the length of the summary by setting the maximum length parameter during generation. This ensures that the summary remains concise while still capturing the essential points.

### Performance Metrics and Evaluation

See test results: [evaluation.log](reports%2Fevaluation.log) and [README.md](reports%2FREADME.md)

To evaluate the performance of each function in the application, we can use the following metrics:

1. **Write_properly:** We can use metrics such as BLEU score, METEOR, or F1 score to measure the overall quality of the improved text compared to a human-written reference. We can also conduct human evaluations to assess the perceived improvement in grammar and style.

2. **Write_the_same_grammar_fixed:** We can use metrics such as grammatical error rate (GER) or F1 score for grammatical error correction to measure the accuracy of the grammatical corrections. Human evaluations can also be conducted to assess the correctness and naturalness of the corrected text.

3. **Summarize:** We can use metrics such as ROUGE score or F1 score to measure the similarity between the generated summary and a human-written reference summary. Human evaluations can also be conducted to assess the informativeness, coherence, and readability of the summaries.

To gather feedback or data to improve the system continuously, we can implement user feedback mechanisms, such as surveys or rating systems, to collect user opinions on the generated outputs. We can also analyze the user interactions and usage patterns to identify areas for improvement.

Prompt engineering plays a crucial role in improving the system's performance. By carefully crafting the prompts, we can guide the models to generate more accurate and appropriate outputs. We can also use prompt engineering to explore different variations of the tasks and experiment with different approaches to achieve the best results.

## Why Should One Use RAG?
There are three ways an LLM can learn new data.

### Training
A large mesh of neural networks is trained over trillions of tokens with billions of parameters to create Large Language Models. The parameters of a deep learning model are the coefficients or weights that hold all the information regarding the particular model. To train a model like GPT-4 costs hundreds of millions of dollars. This way is beyond anyone’s capacity. We cannot re-train such a humongous model on new data. This is not feasible.
### Fine-tuning
Another option is to fine-tune a model on existing data. Fine-tuning involves using a pre-trained model as a starting point during training. We use the knowledge of the pre-trained model to train a new model on different data sets. Albeit it is very potent, it is expensive in terms of time and money. Unless there is a specific requirement, fine-tuning does not make sense.
### Prompting
Prompting is the method where we fit new information within the context window of an LLM and make it answer the queries from the information given in the prompt. It may not be as effective as knowledge learned during training or fine-tuning, but it is sufficient for many real-life use cases, such as document Q&A.

### RAG
Prompting for answers from text documents is effective, but these documents are often much larger than the context windows of Large Language Models (LLMs), posing a challenge. Retrieval Augmented Generation (RAG) pipelines address this by processing, storing, and retrieving relevant document sections, allowing LLMs to answer queries efficiently. So, let’s discuss the crucial components of an RAG pipeline.

## What Are The RAG Components?
In a typical RAG process, we have a few components.

### Text Splitter
Splits documents to accommodate context windows of LLMs.
### Embedding Model
The deep learning model used to get embeddings of documents.
### Vector Stores
The databases where document embeddings are stored and queried along with their metadata.
### LLM
The Large Language Model responsible for generating answers from queries.
Utility Functions: This involves additional utility functions such as Webretriver and document parsers that aid in retrieving and pre-processing files.

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │
    ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
