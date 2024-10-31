from argparse import ArgumentParser
import logging
import random
import sys
from typing import List, Tuple

from datasets import load_dataset
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

from utils.utils import calculate_similarity


ds_analyser_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42


def get_relevance_pairs(questions: List[str], documents: List[str]) -> List[Tuple[str, str]]:
    return list(zip(questions, documents))


def get_irrelevance_pairs(questions: List[str], documents: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    set_of_indices = set(range(len(questions)))
    for idx, val in enumerate(questions):
        other_idx = random.choice(list(set_of_indices - {idx}))
        pairs.append((val, documents[other_idx]))
    return pairs


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

    similarities_for_relevance_pairs = [
        calculate_similarity(it[0], it[1], tokenizer, model)
        for it in tqdm(get_relevance_pairs(ds['train']['question'], ds['train']['context']))
    ]
    similarities_for_irrelevance_pairs = [
        calculate_similarity(it[0], it[1], tokenizer, model)
        for it in tqdm(get_irrelevance_pairs(ds['train']['question'], ds['train']['context']))
    ]
    info_msg = (f'Training data similarities: relevance = {round(np.mean(similarities_for_relevance_pairs), 6)} += '
                f'{np.std(similarities_for_relevance_pairs)}')
    ds_analyser_logger.info(info_msg)
    info_msg = (f'Training data similarities: irrelevance = {round(np.mean(similarities_for_irrelevance_pairs), 6)} += '
                f'{np.std(similarities_for_irrelevance_pairs)}')
    ds_analyser_logger.info(info_msg)

    similarities_for_relevance_pairs = [
        calculate_similarity(it[0], it[1], tokenizer, model)
        for it in tqdm(get_relevance_pairs(ds['validation']['question'], ds['validation']['context']))
    ]
    similarities_for_irrelevance_pairs = [
        calculate_similarity(it[0], it[1], tokenizer, model)
        for it in tqdm(get_irrelevance_pairs(ds['validation']['question'], ds['validation']['context']))
    ]
    info_msg = (f'Validation data similarities: relevance = {round(np.mean(similarities_for_relevance_pairs), 6)} += '
                f'{np.std(similarities_for_relevance_pairs)}')
    ds_analyser_logger.info(info_msg)
    info_msg = (f'Validation data similarities: irrelevance = {round(np.mean(similarities_for_irrelevance_pairs), 6)} += '
                 f'{np.std(similarities_for_irrelevance_pairs)}')
    ds_analyser_logger.info(info_msg)

    similarities_for_relevance_pairs = [
        calculate_similarity(it[0], it[1], tokenizer, model)
        for it in tqdm(get_relevance_pairs(ds['test']['question'], ds['test']['context']))
    ]
    similarities_for_irrelevance_pairs = [
        calculate_similarity(it[0], it[1], tokenizer, model)
        for it in tqdm(get_irrelevance_pairs(ds['test']['question'], ds['test']['context']))
    ]
    info_msg = (f'Test data similarities: relevance = {round(np.mean(similarities_for_relevance_pairs), 6)} += '
                f'{np.std(similarities_for_relevance_pairs)}')
    ds_analyser_logger.info(info_msg)
    info_msg = (f'Test data similarities: irrelevance = {round(np.mean(similarities_for_irrelevance_pairs), 6)} += '
                 f'{np.std(similarities_for_irrelevance_pairs)}')
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
