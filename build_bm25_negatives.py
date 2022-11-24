import os
import json
import argparse
from tqdm import tqdm

from utils.indexing_utils import SparseIndexer, DocumentCollection
from utils import get_logger


logger = get_logger(__name__)


def construct_reverse_passage_mapper(collection, logging_step=10000):
    reverse_mapper = {}
    all_length = len(collection)
    logger.info(f"Constructing reverse pid mapper fisrt! Total number of passages: {all_length}")
    for idx in range(all_length):
        pid = collection.get_pid(idx)
        reverse_mapper[pid] = idx
        if idx % logging_step == 0:
            logger.info(f"{idx}/{all_length} ...")
    return reverse_mapper


def read_orconvqa_data(dataset, read_by="all_questions", is_test=False):
    examples = []
    for idx, data in tqdm(enumerate(dataset)):
        context = []
        for idx, q in enumerate(data["history"]):
            text = q["question"]
            context.append(text)
        
        guid = data["qid"]
        target_question = data["question"]
        truth_answer = ""
        
        if read_by == "all_questions":
            x = context + [target_question]
            x = " ".join(x)
        elif read_by == "all_questions_without_this":
            x = " ".join(context)

        elif read_by == "original":
            x = target_question
        else:
            raise Exception("Unsupported option!")

        examples.append([guid, x])
        if is_test:
            logger.info(f"{guid}: {x}")
            if len(examples) == 10:
                break
    return examples


def read_qrecc_data(dataset, read_by="all", is_test=False):
    context_map = {}
    prev_id = None
    examples = []
    for data in tqdm(dataset):
        guid = f"{data['Conversation_no']}_{data['Turn_no']}"
        did = data["Conversation_no"]
        context = context_map.get(did, [])
        assert len(context) % 2 == 0

        if not context:
            context_map[did] = []

        target_question = data["Question"]
        
        if read_by == "all":
            x = context + [target_question]
            x = " ".join(x)
        elif read_by == "all_without_this":
            x = context
            x = " ".join(x)
        elif read_by == "rewrite":
            x = data["Truth_rewrite"]
        elif read_by == "original":
            x = data["Question"]
        elif read_by == "previous_answer":
            if context:
                pa = context[-1]
            else:
                pa = ""
            x = [pa, data["Question"]]
            x = " ".join(x)
        elif read_by == "previous_answer_only":
            if context:
                pa = context[-1]
            else:
                pa = ""
            x = pa
        elif read_by == "this_answer":
            x = [data["Question"], data["Truth_answer"]]
            x = " ".join(x)
        elif read_by == "this_answer_only":
            x = data["Truth_answer"]
        else:
            raise Exception("Unsupported option!")

        examples.append([guid, x])
        
        context_map[did].append(data["Question"])
        context_map[did].append(data["Truth_answer"])
        
        if is_test:
            logger.info(f"{guid}: {x}")
            if len(examples) == 10:
                break
        
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--read_by', type=str, default="all")
    parser.add_argument('--raw_data_path', type=str, default=None)
    parser.add_argument('--preprocessed_data_path', type=str, default=None)
    parser.add_argument('--pyserini_index_path', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=100)
    args = parser.parse_args()

    if "qrecc" in args.task:
        k_1 = 0.82
        b = 0.68
    else:
        k_1 = 0.9
        b = 0.4

    indexer = SparseIndexer(args.pyserini_index_path)
    indexer.set_retriever(k_1, b)

    if args.task == "orconvqa":
        data = []
        for line in open(f"{args.raw_data_path}/{args.task}/{args.split}.json", "r", encoding="utf-8"):
            data.append(json.loads(line))
        raw_examples = read_orconvqa_data(data, args.read_by)
    else:
        data = json.load(open(f"{args.raw_data_path}/{args.task}/{args.split}.json", "r", encoding="utf-8"))
        raw_examples = read_qrecc_data(data, args.read_by)
        
    if not os.path.exists(f"{args.preprocessed_data_path}/{args.task}/reverse_pids.json"):
        collection = DocumentCollection(f"{args.preprocessed_data_path}/{args.task}/test_collections/data.h5")
        revserse_pids = construct_reverse_passage_mapper(collection)
        json.dump(reverse_pids, open(f"{args.preprocessed_data_path}/{args.task}/reverse_pids.json", "w", encoding="utf-8"))
    else:
        reverse_pids = json.load(open(f"{args.preprocessed_data_path}/{args.task}/reverse_pids.json", "r", encoding="utf-8"))

    scores = {}
    indices = {}
    for idx, line in enumerate(raw_examples):
        qid, q = line
        if not q:
            scores[qid] = {}
            continue

        retrieved_passages = indexer.retrieve(q, args.top_k)
        score = {}
        index = []
        for passage in retrieved_passages:
            score[passage["id"]] = passage["score"]
            index.append(reverse_pids[passage["id"]])
        scores[qid] = score
        indices[qid] = index
        logger.info(f"{idx}/{len(raw_examples)}")

    json.dump(
        scores,
        open(os.path.join(args.preprocessed_data_path, f"{args.split}_bm25_scores.json"), "w"),
        indent=4
    )
    json.dump(
        indices,
        open(os.path.join(args.preprocessed_data_path, f"{args.split}_bm25_indices.json"), "w"),
        indent=4
    )

    examples = load_processed_data(f"{args.preprocessed_data_path}/{args.task}/{args.split}.json")
    logger.info(f"Hard Negatives Mining is done")
    with open(os.path.join(args.preprocessed_data_path, f"{args.split}_bm25_negs.json"), "w") as f:
        for example in examples:
            negative_ids = indices[example.guid]
            negative_scores = scores[example.guid]
            example.hard_negative_ids = negative_ids
            example.hard_negative_scores = negative_scores
            f.write(json.dumps(example.to_dict()) + "\n")
