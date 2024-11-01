from argparse import ArgumentParser
import codecs
import csv
import logging
import os
import random
import sys
from typing import List, Tuple

from datasets import load_dataset
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

from utils.utils import calculate_similarity


ds_preparation_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42


def get_relevance_pairs(questions: List[str], documents: List[str]):
    for it in zip(questions, documents):
        yield it


def get_irrelevance_pairs(questions: List[str], documents: List[str],
                          tokenizer: RobertaTokenizer, model: RobertaModel):
    set_of_indices = set(range(len(questions)))
    for idx, val in enumerate(tqdm(questions)):
        other_indices = random.sample(population=list(set_of_indices - {idx}), k=20)
        similarities = [calculate_similarity(questions[idx], documents[doc_idx], tokenizer, model)
                        for doc_idx in other_indices]
        indices_and_similarities = sorted(
            list(zip(other_indices, similarities)),
            key=lambda x: (-x[1], x[0])
        )
        yield val, documents[indices_and_similarities[0][0]]
        yield val, documents[indices_and_similarities[1][0]]


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        ds_preparation_logger.error(err_msg)
        raise ValueError(err_msg)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_dataset_name', type=str, required=True,
                        help='The input dataset for RAG.')
    parser.add_argument('-o', '--output', dest='output_dataset_name', type=str, required=True,
                        help='The output dataset for RAG.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=False,
                        default='ai-forever/ru-en-RoSBERTa', help='The RoSBERTa name for embeddings.')
    args = parser.parse_args()

    output_dataset_dir = os.path.normpath(args.output_dataset_name)
    if not os.path.isdir(output_dataset_dir):
        err_msg = f'The directory "{output_dataset_dir}" does not exist!'
        ds_preparation_logger.error(err_msg)
        raise IOError(err_msg)

    try:
        ds = load_dataset(args.input_dataset_name)
    except Exception as err:
        ds_preparation_logger.error(str(err))
        raise

    info_msg = (f'The dataset is loaded from "{args.input_dataset_name}". '
                f'There are {len(ds["train"])} samples for training, '
                f'{len(ds["validation"])} samples for validation and {len(ds["test"])} samples for testing.')
    ds_preparation_logger.info(info_msg)

    try:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    except Exception as err:
        ds_preparation_logger.error(str(err))
        raise
    try:
        model = RobertaModel.from_pretrained(args.model_name).to('cuda:0')
    except Exception as err:
        ds_preparation_logger.error(str(err))
        raise
    model.eval()

    text_template = 'Есть ли ответ на вопрос в этом документе?\n\nВопрос: {question}\n\nДокумент: {doc}'
    positive_answer = 'Да, ответ на вопрос есть в этом документе.'
    negative_answer = 'Нет, ответа на вопрос нет в этом документе.'

    with codecs.open(os.path.join(output_dataset_dir, 'validation_data.csv'), mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['text', 'category', 'label'])
        for cur_question, cur_doc in get_relevance_pairs(ds['validation']['question'], ds['validation']['context']):
            data_writer.writerow([text_template.format(question=cur_question, doc=cur_doc), positive_answer, '1'])
        for cur_question, cur_doc in get_irrelevance_pairs(ds['validation']['question'], ds['validation']['context'],
                                                           tokenizer, model):
            data_writer.writerow([text_template.format(question=cur_question, doc=cur_doc), negative_answer, '0'])

    with codecs.open(os.path.join(output_dataset_dir, 'test_data.csv'), mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['text', 'category', 'label'])
        for cur_question, cur_doc in get_relevance_pairs(ds['test']['question'], ds['test']['context']):
            data_writer.writerow([text_template.format(question=cur_question, doc=cur_doc), positive_answer, '1'])
        for cur_question, cur_doc in get_irrelevance_pairs(ds['test']['question'], ds['test']['context'],
                                                           tokenizer, model):
            data_writer.writerow([text_template.format(question=cur_question, doc=cur_doc), negative_answer, '0'])

    with codecs.open(os.path.join(output_dataset_dir, 'training_data.csv'), mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['text', 'category', 'label'])
        for cur_question, cur_doc in get_relevance_pairs(ds['train']['question'], ds['train']['context']):
            data_writer.writerow([text_template.format(question=cur_question, doc=cur_doc), positive_answer, '1'])
        for cur_question, cur_doc in get_irrelevance_pairs(ds['train']['question'], ds['train']['context'],
                                                           tokenizer, model):
            data_writer.writerow([text_template.format(question=cur_question, doc=cur_doc), negative_answer, '0'])


if __name__ == '__main__':
    ds_preparation_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    ds_preparation_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('nerag_dataset_preparation.log')
    file_handler.setFormatter(formatter)
    ds_preparation_logger.addHandler(file_handler)
    main()
