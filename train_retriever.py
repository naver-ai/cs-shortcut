import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import DPRQuestionEncoder, DPRContextEncoder
import argparse
from tqdm import tqdm
import numpy as np
import random
import json
import time
import os
import shutil
import sys
import logging

import pytrec_eval
from utils import retrieval_utils, indexing_utils, set_seed, get_logger, batch_to_device
from utils.distributed_utils import is_main, init_distributed, get_world_size, get_rank
from utils.distributed_utils import dist_print, data_sharding, all_gather_items
from utils.indexing_utils import DenseIndexer, data_sharding, DocumentCollection

from data.base import load_processed_data
from dpr import DPRForPretraining, dpr_init_from_bert


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger(__name__)


def evaluation(eval_loader,
               model,
               indexer,
               qrel_path,
               top_k=100,
               logging_step=10,
               world_size=1,
               rank=0,
               max_buffer_size=592000,
               filtering_missing=True):
    model.eval()
    all_score = {}
    all_index = {}
    
    while len(eval_loader) < logging_step:
        if logging_step <= 1:
            logging_step = 1
            break

        logging_step = int(logging_step / 10)

    dist_print("Evaluation start!", logger)
    
    for idx, batch in enumerate(eval_loader):
        all_input_ids, all_input_masks, all_ids = batch["input_ids"], batch["input_masks"], batch["ids"]
        input_ids = data_sharding(all_input_ids, world_size, rank)
        input_masks = data_sharding(all_input_masks, world_size, rank)
        local_ids = data_sharding(all_ids, world_size, rank)

        with torch.no_grad():
            local_outputs = model.q_encoder(input_ids.to(device),
                                            input_masks.to(device))
            local_outputs = local_outputs.pooler_output
            
        global_outputs, global_ids = all_gather_items([local_outputs, local_ids],
                                                      world_size,
                                                      rank,
                                                      max_buffer_size)

        if is_main():
            global_outputs = global_outputs.cpu().detach().numpy()
            if -1 in global_ids:
                global_ids = [i for i in global_ids if i >= 0]
                global_outputs = global_outputs[:len(global_ids)]

            score_dict, index_dict = indexer.retrieve(global_outputs, global_ids, top_k)
            all_score.update(score_dict)
            all_index.update(index_dict)
        
        if dist.is_initialized():
            dist.barrier(device_ids=[args.local_rank])
        
        if is_main() and idx % logging_step == 0:
            dist_print(f"[{idx}/{len(eval_loader)}]", logger)

    if is_main():
        with open(qrel_path) as handle:
            qrels = json.load(handle)
            if filtering_missing:
                # QReCC: filtering missings
                qrels = dict(filter(lambda x: x[1] != {"": 1}, qrels.items()))

        evaluator = pytrec_eval.RelevanceEvaluator(
                qrels, {'recip_rank', 'recall', 'map'})
        metrics = evaluator.evaluate(all_score)
        map_list = [v['map'] for v in metrics.values()]
        mrr_list = [v['recip_rank'] for v in metrics.values()]
        recall_5_list = [v['recall_5'] for v in metrics.values()]
        recall_10_list = [v['recall_10'] for v in metrics.values()]
        recall_20_list = [v['recall_20'] for v in metrics.values()]
        recall_100_list = [v['recall_100'] for v in metrics.values()]
        eval_metrics = {
            'MAP': np.average(map_list),
            'MRR': np.average(mrr_list),
            'Recall@5': np.average(recall_5_list),
            'Recall@10': np.average(recall_10_list),
            'Recall@20': np.average(recall_20_list),
            'Recall@100': np.average(recall_100_list)
        }
    else:
        eval_metrics = {}
    
    return eval_metrics, all_score, all_index


def training(args,
             num_train_epochs,
             model,
             optimizer,
             scheduler,
             train_loader,
             valid_loader):
    scaler = GradScaler()
    n_slide = args.n_hard_negative + 1
    best_epoch = 0
    best_score = -1
    global_step = 0
    for epoch in range(num_train_epochs):
        model.train()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        for idx, batch in enumerate(train_loader):
            batch = batch_to_device(batch, device)
            input_ids, input_masks, labels = batch["input_ids"], batch["input_masks"], batch["labels"]
            batch_size = input_ids.size(0)
            positive_idx = np.arange(0, batch_size * n_slide, n_slide).tolist()
            hnegative_idx = []
            if n_slide > 1:
                cand_ids = torch.cat([batch["cand_ids"][torch.arange(batch_size), labels[:,i]].unsqueeze(1) 
                                      for i in range(n_slide)], 1)
                cand_masks = torch.cat([batch["cand_masks"][torch.arange(batch_size), labels[:,i]].unsqueeze(1) 
                                        for i in range(n_slide)], 1)
                if args.pseudo_positive_ratio:
                    hnegative_idx = (np.arange(0, batch_size * n_slide, n_slide) + 1).tolist()
            else:
                cand_ids = batch["cand_ids"][torch.arange(batch_size), labels]
                cand_masks = batch["cand_masks"][torch.arange(batch_size), labels]

            if len(cand_ids.size()) == 2:
                cand_ids = cand_ids.unsqueeze(1)
                cand_masks = cand_masks.unsqueeze(1)
            optimizer.zero_grad()

            with autocast():                    
                loss, prob = model(input_ids,
                                   input_masks,
                                   cand_ids,
                                   cand_masks,
                                   positive_idx,
                                   hnegative_idx)

            if args.n_gpu > 1:
                loss = loss.mean()

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if idx % 100 == 0:
                msg = f"[{epoch}/{num_train_epochs}][{idx}/{len(train_loader)}] loss: {loss.item()}"
                dist_print(msg, logger)
        
        model_to_eval = model.module if hasattr(model, 'module') else model
        indexer = DenseIndexer(os.path.join(args.data_path, args.task, "dev_collections"),
                               batch_size=args.index_batch_size, max_buffer_size=args.max_buffer_size, logger=logger)
        indexer.set_collections()
        indexer.passage_inference(model_to_eval.ctx_encoder,
                                  os.path.join(args.output_path, "index_dev.faiss"),
                                  args.local_rank,
                                  get_world_size())
        # TODO
        if os.path.exists(os.path.join(args.data_path, args.task, 'qrels_dev.txt')):
            qrel_path = os.path.join(args.data_path, args.task, 'qrels_dev.txt')
        else:
            qrel_path = args.qrel_path
        eval_result, scores, indices = evaluation(valid_loader,
                                                  model_to_eval,
                                                  indexer,
                                                  qrel_path,
                                                  args.top_k,
                                                  world_size=get_world_size(),
                                                  rank=get_rank(),
                                                  max_buffer_size=args.max_buffer_size)
        if is_main():
            dist_print(f"Epoch {epoch}, {eval_result}", logger)
            if eval_result['Recall@5'] >= best_score:
                best_score = eval_result['Recall@5']
                model_to_eval.save_pretrained(args.output_path)
                json.dump(
                    eval_result,
                    open(os.path.join(args.output_path, "dev_eval_result.json"), "w"),
                    indent=4
                )
                json.dump(
                    scores,
                    open(os.path.join(args.output_path, "dev_eval_scores.json"), "w"),
                    indent=4
                )
                json.dump(
                    indices,
                    open(os.path.join(args.output_path, "dev_eval_indices.json"), "w"),
                    indent=4
                )
                dist_print(f"Save the best model...", logger)

        if dist.is_initialized():
            dist.barrier(device_ids=[args.local_rank])

    return model, best_epoch, best_score


def load_model(args,
               tokenizer,
               num_train_epochs,
               init_from_previous=False,
               only_model=False):
    
    if init_from_previous:
        model = DPRForPretraining.from_pretrained(args.model_name_or_path)
        model.max_buffer_size = args.max_buffer_size
    else:
        if "bert" in args.model_name_or_path:
            q_encoder = dpr_init_from_bert(DPRQuestionEncoder, args.model_name_or_path)
            ctx_encoder = dpr_init_from_bert(DPRContextEncoder, args.model_name_or_path)
        else:
            q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        model = DPRForPretraining(q_encoder, ctx_encoder, max_buffer_size=args.max_buffer_size)
        model.resize_token_embeddings(len(tokenizer))

    if args.tie_encoder:
        model.tie_encoder()
    model.to(device)
    
    if only_model:
        return model

    t_total = args.n_steps * num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.num_warmup_steps:
        num_warmup_steps = args.num_warmup_steps
    else:
        num_warmup_steps = int(t_total * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
            )
    
    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    return model, optimizer, scheduler
    

def main(args):
    set_seed(args.random_seed)
    args = init_distributed(args)
    rng = random.Random(args.random_seed)
    
    q_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    ctx_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    negative_sampler = None
    if args.n_hard_negative:
        negative_sampler = DocumentCollection(os.path.join(args.data_path, args.task, "test_collections", "data.h5"))
    if "/" not in args.train_data:
        train_examples = load_processed_data(os.path.join(args.data_path, args.task, args.train_data))
    else:
        train_examples = load_processed_data(args.train_data)

    train_data = retrieval_utils.RetrievalDataset(
        train_examples,
        sampler=negative_sampler,
        n_negative=args.n_hard_negative,
        pad_token_id=q_tokenizer.pad_token_id,
        tokenizer=q_tokenizer,
        rng=rng
    )

    if args.distributed:
        train_sampler = DistributedSampler(train_data, shuffle=True)
        train_sampler.set_epoch(0)
    else:
        train_sampler = RandomSampler(train_data)

    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              collate_fn=train_data.collate_fn)
    args.n_steps = len(train_loader)

    valid_examples = load_processed_data(os.path.join(args.data_path, args.task, args.dev_data))
    valid_data = retrieval_utils.RetrievalDataset(valid_examples)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.eval_batch_size * get_world_size(),
                              shuffle=False,
                              collate_fn=valid_data.collate_fn)
    
    test_examples = load_processed_data(os.path.join(args.data_path, args.task, args.test_data))
    test_data = retrieval_utils.RetrievalDataset(test_examples)
    test_loader = DataLoader(test_data,
                             batch_size=args.eval_batch_size * get_world_size(),
                             shuffle=False,
                             collate_fn=test_data.collate_fn)
    
    if is_main() and os.path.exists(args.output_path):
        q_tokenizer.save_pretrained(args.output_path)
        json.dump(vars(args), open(os.path.join(args.output_path, "exp.json"), "w"), indent=4)

    if not args.index_data_path:
        args.index_data_path = args.output_path
    
    args.qrel_path = os.path.join(args.data_path, args.task, 'qrels.txt')
    model, optimizer, scheduler = load_model(args,
                                             q_tokenizer,
                                             num_train_epochs=args.num_train_epochs,
                                             init_from_previous=False)
    dist_print(f"training with batch size: {args.train_batch_size * args.n_all_gpu}", logger)

    model, best_epoch, best_score = training(args,
                                             args.num_train_epochs,
                                             model,
                                             optimizer,
                                             scheduler,
                                             train_loader,
                                             valid_loader
                                            )

    if not args.do_predict:
        return
    
    dist_print(f"Indexing & Evaluaiton start. It takes a lot of time since it is based on exact search...", logger)
    
    # Evaluation on test set
    args.model_name_or_path = args.output_path
    model = load_model(args,
                       q_tokenizer,
                       num_train_epochs=args.num_train_epochs,
                       init_from_previous=True,
                       only_model=True)
    
    indexer = DenseIndexer(os.path.join(args.data_path, args.task, "test_collections"),
                           batch_size=args.index_batch_size, max_buffer_size=args.max_buffer_size, logger=logger)
    indexer.set_collections()
    indexer.passage_inference(model.ctx_encoder,
                              os.path.join(args.output_path, "index_test.faiss"),
                              args.local_rank,
                              get_world_size())
    
    eval_result, scores, indices = evaluation(test_loader,
                                              model,
                                              indexer,
                                              args.qrel_path,
                                              args.top_k,
                                              world_size=get_world_size(),
                                              rank=get_rank(),
                                              max_buffer_size=args.max_buffer_size)
    dist_print(f"Test score: {eval_result}", logger)
    
    if is_main():
        json.dump(
            eval_result,
            open(os.path.join(args.output_path, "test_eval_result.json"), "w"),
            indent=4
        )
        json.dump(
            scores,
            open(os.path.join(args.output_path, "test_eval_scores.json"), "w"),
            indent=4
        )
        json.dump(
            indices,
            open(os.path.join(args.output_path, "test_eval_indices.json"), "w"),
            indent=4
        )
    if dist.is_initialized():
        dist.barrier(device_ids=[args.local_rank])

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--train_data', type=str, default='train.json')
    parser.add_argument('--dev_data', type=str, default='dev.json')
    parser.add_argument('--test_data', type=str, default='test.json')
    parser.add_argument('--output_path', type=str, default="outputs")
    parser.add_argument('--task', type=str, default='orconvqa')
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=2.0)
    parser.add_argument("--distributed", action='store_true', default=False,
                        help="training with DDP")
    parser.add_argument("--local_rank", type=int, default=0)
    
    parser.add_argument('--n_hard_negative', type=int, default=0)
    parser.add_argument('--pseudo_positive_ratio', type=float, default=0.0)

    parser.add_argument('--tie_encoder', action='store_true', default=False)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--index_batch_size', type=int, default=64)
    parser.add_argument('--index_data_path', type=str, default=None)
    parser.add_argument('--qrel_path', type=str, default=None)
    parser.add_argument('--max_buffer_size', type=int, default=592000)
    parser.add_argument('--do_predict', action='store_true', default=False)
    args = parser.parse_args()

    if is_main() and not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    fileHandler = logging.FileHandler(f"{args.output_path}/log.out", "a")
    formatter = logging.Formatter('%(asctime)s > %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(args.output_path)
    logger.info("logging start!")
    main(args)
