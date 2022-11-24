import json
import os
import sys
import logging
import argparse
import shutil
from glob import glob

from pyserini.index import IndexReader
from transformers import BertTokenizerFast

from utils import set_seed, get_logger
from data.orconvqa import OrConvQAProcessor
from data.qrecc import QReCCProcessor
from utils.indexing_utils import DocumentCollection

logger = get_logger(__name__)


def main(args):
    set_seed(args.random_seed)
    q_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    ctx_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
        
    if not os.path.exists(os.path.join(args.output_path, args.task)):
        os.mkdir(os.path.join(args.output_path, args.task))

    if args.data_path:
        
        if "orconvqa" in args.task:
            processor = OrConvQAProcessor(os.path.join(args.data_path, args.task),
                                          q_tokenizer,
                                          ctx_tokenizer,
                                          args.max_query_length,
                                          args.max_passage_length,
                                          args.retain_first_question,
                                          args.use_only_queries,
                                          args.use_rewrite_only,
                                          verbose=True,
                                          logger=logger)

        elif "qrecc" in args.task:
            try:
                index_reader = IndexReader(args.pyserini_index_path)
            except:
                raise Exception("Please build pyserini index first!")
            processor = QReCCProcessor(os.path.join(args.data_path, args.task),
                                       q_tokenizer,
                                       ctx_tokenizer,
                                       index_reader,
                                       args.max_query_length,
                                       args.max_passage_length,
                                       args.retain_first_question,
                                       args.use_only_queries,
                                       args.use_rewrite_only,
                                       verbose=True,
                                       logger=logger)
            shutil.copyfile(f"{args.data_path}/{args.task}/test_question_types.json", f"{args.output_path}/{args.task}/test_question_types.json")
            shutil.copyfile(f"{args.data_path}/{args.task}/qrels_dev.txt", f"{args.output_path}/{args.task}/qrels_dev.txt")

        for split in ["train", "dev", "test"]:
            examples = processor.read_examples(f"{split}.json",
                                               skip_no_truth_passages=False)

            if args.suffix:
                split = f"{split}_{args.suffix}"

            with open(os.path.join(args.output_path, args.task, f"{split}.json"), "w", encoding="utf-8") as f:
                for example in examples:
                    f.write(json.dumps(example.to_dict()) + "\n")

            logger.info(f"Save {args.task} / {split}")
            if "train" in split and "qrecc" in args.task:
                examples = list(filter(lambda x: x.has_positive, examples))
                with open(os.path.join(args.output_path, args.task, f"{split}_filtered.json"), "w", encoding="utf-8") as f:
                    for example in examples:
                        f.write(json.dumps(example.to_dict()) + "\n")
                logger.info(f"Save {args.task} / {split}_filtered")

            if args.train_only:
                break
                
    shutil.copyfile(f"{args.data_path}/{args.task}/qrels.txt", f"{args.output_path}/{args.task}/qrels.txt")
                
    if args.dev_collection_path:
        passage_files = glob(f"{args.dev_collection_path}")
        logger.info(f"Overall {len(passage_files)} documents")
        
        if not os.path.exists(os.path.join(args.output_path, args.task, "dev_collections")):
            os.mkdir(os.path.join(args.output_path, args.task, "dev_collections"))
        
        output_path = os.path.join(args.output_path, args.task, "dev_collections", "data.h5")
        collection = DocumentCollection(output_path, max_passage_length=args.max_passage_length)
        collection.write_h5(passage_files, ctx_tokenizer)
    
    if args.test_collection_path:
        passage_files = glob(f"{args.test_collection_path}")
        logger.info(f"Overall {len(passage_files)} documents")
        
        if not os.path.exists(os.path.join(args.output_path, args.task, "test_collections")):
            os.mkdir(os.path.join(args.output_path, args.task, "test_collections"))
        
        output_path = os.path.join(args.output_path, args.task, "test_collections", "data.h5")
        collection = DocumentCollection(output_path, max_passage_length=args.max_passage_length)
        collection.write_h5(passage_files, ctx_tokenizer)

    config = vars(args)
    with open(os.path.join(args.output_path, args.task, f"config_{args.suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--max_query_length', type=int, default=128)
    parser.add_argument('--max_passage_length', type=int, default=384)
    parser.add_argument('--retain_first_question', action='store_true', default=False)
    parser.add_argument('--use_only_queries', action='store_true', default=False)
    parser.add_argument('--use_rewrite_only', action='store_true', default=False)
    parser.add_argument('--exclude_current_question', action='store_true', default=False)
    parser.add_argument('--train_only', action='store_true', default=False)
    parser.add_argument('--test_collection_path', type=str, default="")
    parser.add_argument('--dev_collection_path', type=str, default="")
    parser.add_argument('--pyserini_index_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
