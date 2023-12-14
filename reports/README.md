# Performance Metrics and Evaluation

To evaluate the performance of each function in the application, we can use the following metrics:

1. **Write_properly:** We can use metrics such as BLEU score, METEOR, or F1 score to measure the overall quality of the improved text compared to a human-written reference. We can also conduct human evaluations to assess the perceived improvement in grammar and style.

2. **Write_the_same_grammar_fixed:** We can use metrics such as grammatical error rate (GER) or F1 score for grammatical error correction to measure the accuracy of the grammatical corrections. Human evaluations can also be conducted to assess the correctness and naturalness of the corrected text.

3. **Summarize:** We can use metrics such as ROUGE score or F1 score to measure the similarity between the generated summary and a human-written reference summary. Human evaluations can also be conducted to assess the informativeness, coherence, and readability of the summaries.

To gather feedback or data to improve the system continuously, we can implement user feedback mechanisms, such as surveys or rating systems, to collect user opinions on the generated outputs. We can also analyze the user interactions and usage patterns to identify areas for improvement.

Prompt engineering plays a crucial role in improving the system's performance. By carefully crafting the prompts, we can guide the models to generate more accurate and appropriate outputs. We can also use prompt engineering to explore different variations of the tasks and experiment with different approaches to achieve the best results.

# Metrics

## BLUE
BLEU, or the Bilingual Evaluation Understudy, is a metric for comparing a candidate translation to one or more reference translations. Although developed for translation, it can be used to evaluate text generated for different natural language processing tasks, such as paraphrasing and text summarization.

    pip install git+https://github.com/google-research/bleurt.git

## ROUGE
Recall-Oriented Understudy for Gisting Evaluation, often referred as ROUGE score, is a metric used to evaluate text summarization and translation models. There are variations of ROUGE scores. In this article we will show how to calculate ROUGE-N, ROUGE-L and mention other types of ROUGE scores.

    pip install rouge_score

# Datasets

## JFLEG
The JFLEG (JHU FLuency-Extended GUG) dataset is a comprehensive benchmark for English Grammatical Error Correction
(GEC) systems. It serves as a gold standard for developing and evaluating the effectiveness of GEC systems
in terms of fluency and grammaticality in English texts. The dataset is specifically designed to assess
the native-sounding quality and grammatical precision of written English sentences.

### Dataset
[Local: jfleg.csv](..%2Fdata%2Fjfleg.csv)

## ROSE
This dataset provides how-to articles from wikihow.com and their summaries, written as a coherent paragraph. The dataset itself is available at wikisum.zip, and contains the article, the summary, the wikihow url, and an official fold (train, val, or test). In addition, human evaluation results are available at wikisum-human-eval.zip. It consists of human evaluation of the summary of the Pegasus system, annotators response regarding the difficulty of the task, and words they marked as unknown.

### Dataset
[Local: wikisum.jsonl](..%2Fdata%2Fwikisum.jsonl)
[Source: WikiSumDataset](https://wikisum.s3.amazonaws.com/WikiSumDataset.zip)
