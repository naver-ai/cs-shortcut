"""
https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag/use_own_knowledge_dataset.py
"""

import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import torch
from datasets import Features, Sequence, Value, load_dataset, load_from_disk

import faiss
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)
import os
import argparse
from utils import set_seed, get_logger


logger = get_logger(__name__)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(str(title) if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def embed(documents: dict,
          ctx_encoder: DPRContextEncoder,
          ctx_tokenizer: DPRContextEncoderTokenizerFast,
          split_doc: bool) -> dict:
    """Compute the DPR embeddings of document passages"""
    
    if split_doc:
        documents = split_documents(documents)
        titles = documents["title"]
        texts = documents["text"]
    else:
        titles = [str(t) for t in documents["title"]]
        texts = [str(t) for t in documents["text"]]

    input_ids = ctx_tokenizer(
        titles, texts, truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def build_hf_dataset(file_name,
                     ctx_encoder,
                     ctx_tokenizer,
                     save_dir,
                     batch_size=256,
                     cache_dir=None,
                     split_doc=False):
    """
    The file should be tsv format containing 'id', 'text', 'title' as columns
    """
    if os.path.exists(os.path.join(save_dir, "dataset.arrow")):
        logger.info("Dataset already exists... Loading...")
        dataset = load_from_disk(save_dir)
        logger.info("Dataset is loaded!")
        return dataset

    logger.info(f"Start building hf dataset for encoding knowledges {file_name}")
    dataset = load_dataset(
            "csv", data_files=[file_name],
        split="train", delimiter="\t", column_names=["id", "text", "title"],
        cache_dir=cache_dir
    )
    
    new_features = Features(
        {"id": Value("int32"), "text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
    )  # optional, save as float32 instead of float64 to save space
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, split_doc=args.split_doc),
        batched=True,
        batch_size=batch_size,
        features=new_features,
    )
    dataset.save_to_disk(save_dir)
    logger.info(f"Save to {save_dir}")
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dimension', type=int, default=768)
    parser.add_argument('--store_n', type=int, default=512)
    parser.add_argument('--ef_search', type=int, default=128)
    parser.add_argument('--ef_construction', type=int, default=200)
    parser.add_argument('--split_doc', action='store_true', default=False)
    args = parser.parse_args()
    set_seed(args.random_seed)
    ctx_encoder = DPRContextEncoder.from_pretrained(args.model_name_or_path)
    ctx_encoder.to(device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(args.model_name_or_path)

    dataset = build_hf_dataset(args.file_name,
                               ctx_encoder,
                               ctx_tokenizer,
                               args.save_dir,
                               args.batch_size,
                               args.cache_dir,
                               args.split_doc)

    index = faiss.IndexHNSWFlat(args.dimension, args.store_n, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = args.ef_search
    index.hnsw.efConstruction = args.ef_construction
    logger.info("Add index start!")
    dataset.add_faiss_index("embeddings", custom_index=index)
    save_file_name = f"hnsw_{args.dimension}_{args.store_n}_{args.ef_search}_{args.ef_construction}.faiss"
    dataset.get_index("embeddings").save(os.path.join(args.save_dir, save_file_name))
    logger.info("Add index done!")
