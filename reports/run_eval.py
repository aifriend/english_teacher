from pathlib import Path

import evaluate
from datasets import load_dataset
from datasets import load_metric
from dotenv import load_dotenv, find_dotenv
from pandas import DataFrame

from english_improvement_agent.EnglishImprovementAgentSummary import EnglishImprovementAgentSummary
from english_improvement_agent.EnglishImprovementAgentWrite import EnglishImprovementAgentWrite
from english_improvement_agent.commonsLib import loggerElk

logger = loggerElk(__name__)


def get_csv(df, dataset):
    for case in dataset:
        # Adding the task's prefix to input
        input_text = case["sentence"]
        for correction in case["corrections"]:
            # a few of the cases contain blank strings.
            if input_text and correction:
                df.loc[len(df)] = dict(
                    input=input_text, target=correction)

    unique_df = df.drop_duplicates()
    return unique_df


def eval_blue(eval_name: str, dataset: DataFrame, model, metric):
    logger.Information(f"eval with score {eval_name} for {len(dataset)} examples")

    for index, row in dataset.iterrows():
        prediction = model(row['input'])
        correction = row['target']
        logger.Information(f"with prediction: '{prediction}'")
        logger.Information(f"and correction: '{correction}'")
        metric.add_batch(predictions=[prediction], references=[correction])
    eval_score = metric.compute()

    return eval_score


def eval_write_grammar():
    logger.Information("loading dataset: JFLEG")
    df = DataFrame(columns=["input", "target"])
    eval_dataset = load_dataset("jfleg", split='test[:]')
    df = get_csv(df, eval_dataset)
    eval_df = df.sample(MAX_EVAL_SAMPLES)

    write_service = EnglishImprovementAgentWrite()
    metric = load_metric('bleurt')
    score = eval_blue('vannity', eval_df, write_service.write_properly_vannity, metric)
    logger.Information(
        f"BLUE score for write properly with HF Vannity model: {score}")

    score = eval_blue('coedit', eval_df, write_service.write_properly_coedit, metric)
    logger.Information(
        f"BLUE score for write properly with HF Coedit model: {score}")

    score = eval_blue('instruct', eval_df, write_service.write_properly_instruct, metric)
    logger.Information(
        f"BLUE score for write properly with OpenAI Instruct model: {score}")


def get_dict(df, dataset):
    for case in dataset:
        # Adding the task's prefix to input
        input_text = case["article"]
        summary = case["summary"]
        # a few of the cases contain blank strings.
        if input_text and summary:
            df.loc[len(df)] = dict(
                input=input_text, target=summary)

    unique_df = df.drop_duplicates()
    return unique_df


def eval_rouge(eval_name: str, dataset: DataFrame, model, metric):
    logger.Information(f"eval with score {eval_name} for {len(dataset)} examples")

    for index, row in dataset.iterrows():
        prediction = model(row['input'])
        correction = row['target']
        logger.Information(f"with prediction: '{prediction}'")
        logger.Information(f"and summary: '{correction}'")
        metric.add_batch(predictions=[prediction], references=[correction])
    eval_score = metric.compute()

    return eval_score


def eval_summary():
    logger.Information("loading dataset: ROSE")
    df = DataFrame(columns=["input", "target"])
    eval_dataset = load_dataset(
        'json', data_files='../data/wikisum.jsonl', split='train')
    df = get_dict(df, eval_dataset)
    eval_df = df.sample(MAX_EVAL_SAMPLES)

    summary_service = EnglishImprovementAgentSummary()
    metric = evaluate.load('rouge')
    score = eval_rouge('T5', eval_df, summary_service.summarization_with_t5, metric)
    logger.Information(
        f"ROUGE score for summary with HF T5 model: {score}")


if __name__ == '__main__':
    MAX_EVAL_SAMPLES = 10

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    eval_write_grammar()
    eval_summary()
