from argparse import ArgumentParser
import logging
import random
import sys
from typing import Tuple

from datasets import load_dataset
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

from utils.utils import calculate_similarity


ds_analyser_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42


def select_question_and_document_from_prompt(prompt: str) -> Tuple[str, str]:
    question_prefix = '\nВопрос: '
    question_idx = prompt.find(question_prefix)
    document_prefix = '\nДокумент: '
    document_idx = prompt.find(document_prefix)

    if (question_idx < 0) or (document_idx < 0) or (document_idx <= question_idx):
        err_msg = f'The text cannot be parsed! {prompt}'
        ds_analyser_logger.error(err_msg)
        raise ValueError(err_msg)
    question = prompt[(question_idx + len(question_prefix)):document_idx].strip()
    document = prompt[(document_idx + len(document_prefix)):]
    return question, document


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        ds_analyser_logger.error(err_msg)
        raise ValueError(err_msg)

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset_name', type=str, required=True,
                        help='The analyzed dataset for RAG.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=False,
                        default='ai-forever/ru-en-RoSBERTa', help='The RoSBERTa name for embeddings.')
    args = parser.parse_args()

    try:
        ds = load_dataset(args.dataset_name)
    except Exception as err:
        ds_analyser_logger.error(str(err))
        raise

    info_msg = (f'The dataset is loaded from "{args.dataset_name}". There are {len(ds["train"])} samples for training, '
                f'{len(ds["validation"])} samples for validation and {len(ds["test"])} samples for testing.')
    ds_analyser_logger.info(info_msg)

    try:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    except Exception as err:
        ds_analyser_logger.error(str(err))
        raise
    try:
        model = RobertaModel.from_pretrained(args.model_name).to('cuda:0')
    except Exception as err:
        ds_analyser_logger.error(str(err))
        raise
    model.eval()

    positive_pairs = list(map(
        select_question_and_document_from_prompt,
        ds['train'].filter(lambda it1: it1['label'] == 1)['text']
    ))
    negative_pairs = list(map(
        select_question_and_document_from_prompt,
        ds['train'].filter(lambda it1: it1['label'] != 1)['text']
    ))
    similarities_for_positive_pairs = [
        calculate_similarity(it3[0], it3[1], tokenizer, model)
        for it3 in tqdm(positive_pairs)
    ]
    similarities_for_negative_pairs = [
        calculate_similarity(it3[0], it3[1], tokenizer, model)
        for it3 in tqdm(negative_pairs)
    ]
    info_msg = (f'Training data similarities: positive = {round(np.mean(similarities_for_positive_pairs), 6)} += '
                f'{np.std(similarities_for_positive_pairs)}')
    ds_analyser_logger.info(info_msg)
    info_msg = (f'Training data similarities: negative = {round(np.mean(similarities_for_negative_pairs), 6)} += '
                f'{np.std(similarities_for_negative_pairs)}')
    ds_analyser_logger.info(info_msg)

    positive_pairs = list(map(
        select_question_and_document_from_prompt,
        ds['validation'].filter(lambda it1: it1['label'] == 1)['text']
    ))
    negative_pairs = list(map(
        select_question_and_document_from_prompt,
        ds['validation'].filter(lambda it1: it1['label'] != 1)['text']
    ))
    similarities_for_positive_pairs = [
        calculate_similarity(it3[0], it3[1], tokenizer, model)
        for it3 in tqdm(positive_pairs)
    ]
    similarities_for_negative_pairs = [
        calculate_similarity(it3[0], it3[1], tokenizer, model)
        for it3 in tqdm(negative_pairs)
    ]
    info_msg = (f'Validation data similarities: positive = {round(np.mean(similarities_for_positive_pairs), 6)} += '
                f'{np.std(similarities_for_positive_pairs)}')
    ds_analyser_logger.info(info_msg)
    info_msg = (f'Validation data similarities: negative = {round(np.mean(similarities_for_negative_pairs), 6)} += '
                f'{np.std(similarities_for_negative_pairs)}')
    ds_analyser_logger.info(info_msg)

    positive_pairs = list(map(
        select_question_and_document_from_prompt,
        ds['test'].filter(lambda it1: it1['label'] == 1)['text']
    ))
    negative_pairs = list(map(
        select_question_and_document_from_prompt,
        ds['test'].filter(lambda it1: it1['label'] != 1)['text']
    ))
    similarities_for_positive_pairs = [
        calculate_similarity(it3[0], it3[1], tokenizer, model)
        for it3 in tqdm(positive_pairs)
    ]
    similarities_for_negative_pairs = [
        calculate_similarity(it3[0], it3[1], tokenizer, model)
        for it3 in tqdm(negative_pairs)
    ]
    info_msg = (f'Test data similarities: positive = {round(np.mean(similarities_for_positive_pairs), 6)} += '
                f'{np.std(similarities_for_positive_pairs)}')
    ds_analyser_logger.info(info_msg)
    info_msg = (f'Test data similarities: negative = {round(np.mean(similarities_for_negative_pairs), 6)} += '
                f'{np.std(similarities_for_negative_pairs)}')
    ds_analyser_logger.info(info_msg)


if __name__ == '__main__':
    ds_analyser_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    ds_analyser_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('dataset_analysis.log')
    file_handler.setFormatter(formatter)
    ds_analyser_logger.addHandler(file_handler)
    main()
