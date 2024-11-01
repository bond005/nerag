from typing import List, Union

import numpy as np
from razdel import sentenize
import torch
from transformers import RobertaTokenizer, RobertaModel


def split_long_text(long_text: str, tokenizer: RobertaTokenizer, max_tokens: int = 512) -> List[str]:
    sentences = [it.text for it in sentenize(long_text)]
    paragraphs = []
    old_paragraph_text = ''
    for cur_sent in sentences:
        new_paragraph_text = (old_paragraph_text + ' ' + cur_sent).strip()
        n_tokens = len(tokenizer.tokenize(new_paragraph_text))
        if n_tokens > max_tokens:
            if len(old_paragraph_text) > 0:
                paragraphs.append(old_paragraph_text)
                old_paragraph_text = cur_sent
            else:
                paragraphs.append(new_paragraph_text)
        elif n_tokens == max_tokens:
            paragraphs.append(new_paragraph_text)
            old_paragraph_text = ''
        else:
            old_paragraph_text = new_paragraph_text
    if len(old_paragraph_text) > 0:
        paragraphs.append(old_paragraph_text)
    return paragraphs


def calculate_text_embeddings(text: Union[str, List[str]], prefix: str,
                              tokenizer: RobertaTokenizer, model: RobertaModel) -> np.ndarray:
    if prefix not in {'search_query: ', 'search_document: '}:
        err_msg = f'The prefix is inadmissible! Expected "search_query: " or "search_document: ", got "{prefix}"'
        raise ValueError(err_msg)
    if isinstance(text, str):
        tokenized_inputs = tokenizer(
            [prefix + text], max_length=model.config.max_position_embeddings - 2,
            padding=True, truncation=True, return_tensors='pt'
        ).to(model.device)
    else:
        tokenized_inputs = tokenizer(
            [(prefix + it) for it in text], max_length=model.config.max_position_embeddings - 2,
            padding=True, truncation=True, return_tensors='pt'
        ).to(model.device)
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    embeddings = torch.nn.functional.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1).cpu().numpy()
    return embeddings


def calculate_similarity(question: str, document: str,
                         tokenizer: RobertaTokenizer, model: RobertaModel) -> float:
    paragraphs = split_long_text(long_text=document, tokenizer=tokenizer,
                                 max_tokens=model.config.max_position_embeddings - 2)
    question_vector = calculate_text_embeddings(question, 'search_query: ', tokenizer, model)
    document_vectors = calculate_text_embeddings(paragraphs, 'search_document: ', tokenizer, model)
    similarities = (question_vector @ document_vectors.T).flatten()
    return float(np.max(similarities))
