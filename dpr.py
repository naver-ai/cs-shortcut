import os
import torch
from torch import Tensor, nn
from transformers import BertConfig, BertModel
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from utils.loss import BiEncoderNllLoss
from utils.distributed_utils import get_world_size, all_gather_list, get_rank


class DPRForPretraining(nn.Module):
    def __init__(self, q_encoder, ctx_encoder, max_buffer_size=592000):
        super(DPRForPretraining, self).__init__()
        self.q_encoder = q_encoder
        self.ctx_encoder = ctx_encoder
        self.max_buffer_size = max_buffer_size
        self.loss_fn = BiEncoderNllLoss()
        
    def save_pretrained(self, model_name_or_path):
        if not os.path.exists(model_name_or_path):
            os.mkdir(model_name_or_path)
        self.ctx_encoder.config.vocab_size = self.ctx_encoder.base_model.bert_model.config.vocab_size
        self.ctx_encoder.config.save_pretrained(model_name_or_path)
        torch.save(self.state_dict(), f"{model_name_or_path}/pytorch_model.bin")
        
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        config = DPRConfig.from_pretrained(model_name_or_path)
        q_encoder = DPRQuestionEncoder(config)
        ctx_encoder = DPRContextEncoder(config)
        obj = cls(q_encoder, ctx_encoder)
        obj.load_state_dict(torch.load(f"{model_name_or_path}/pytorch_model.bin", map_location="cpu"))
        return obj
    
    def freeze_ctx_encoder(self):
        for param in self.ctx_encoder.parameters():
            param.requires_grad = False

    def tie_encoder(self):
        self.q_encoder.base_model.bert_model = self.ctx_encoder.base_model.bert_model
    
    def resize_token_embeddings(self, size):
        self.q_encoder.base_model.bert_model.resize_token_embeddings(size)
        self.ctx_encoder.base_model.bert_model.resize_token_embeddings(size)
        
    def forward(self,
                input_ids,
                input_masks,
                cand_ids,
                cand_masks,
                local_positive_idxs,
                local_hard_negatives_idxs,
                is_masked=False):
        b, n_c, l = cand_ids.size()
        local_q_vector = self.q_encoder(input_ids, input_masks).pooler_output
        local_ctx_vectors = self.ctx_encoder(cand_ids.view(b * n_c, -1), cand_masks.view(b * n_c, -1)).pooler_output

        distributed_world_size = get_world_size() or 1
        if distributed_world_size > 1:
            q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
            ctx_vector_to_send = torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()

            global_question_ctx_vectors = all_gather_list(
                [
                    q_vector_to_send,
                    ctx_vector_to_send,
                    local_positive_idxs,
                    local_hard_negatives_idxs,
                ],
                max_size=self.max_buffer_size,
            )

            global_q_vector = []
            global_ctxs_vector = []
            positive_idx_per_question = []
            hard_negatives_per_question = []
            total_ctxs = 0
            for i, item in enumerate(global_question_ctx_vectors):
                q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

                if i != get_rank():
                    global_q_vector.append(q_vector.to(local_q_vector.device))
                    global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                    positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                    hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in hard_negatives_idxs])
                else:
                    global_q_vector.append(local_q_vector)
                    global_ctxs_vector.append(local_ctx_vectors)
                    positive_idx_per_question.extend([v + total_ctxs for v in local_positive_idxs])
                    hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in local_hard_negatives_idxs])
                total_ctxs += ctx_vectors.size(0)
            global_q_vector = torch.cat(global_q_vector, dim=0)
            global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

        else:
            global_q_vector = local_q_vector
            global_ctxs_vector = local_ctx_vectors
            positive_idx_per_question = local_positive_idxs
            hard_negatives_per_question = local_hard_negatives_idxs

        loss, _, softmax_scores = self.loss_fn.calc(global_q_vector,
                                                    global_ctxs_vector,
                                                    positive_idx_per_question,
                                                    hard_negatives_per_question,
                                                    return_only_prob=is_masked)
        return loss, softmax_scores


class DPRForSelectionFromCandidates(DPRForPretraining):
    def __init__(self, q_encoder, ctx_encoder):
        super(DPRForSelectionFromCandidates, self).__init__()
        self.q_encoder = q_encoder
        self.ctx_encoder = ctx_encoder
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, input_masks, cand_ids, cand_masks, labels=None):
        b, n_c, l = cand_ids.size()
        q = self.q_encoder(input_ids, input_masks).pooler_output
        ctx = self.ctx_encoder(cand_ids.view(b * n_c, -1), cand_masks.view(b * n_c, -1))
        ctx = ctx.pooler_output.view(b, n_c, -1)
        score = torch.bmm(ctx, q.unsqueeze(-1)).squeeze(-1) # b, n_c

        outputs = (score, )
        cnt = 0
        if labels is not None:
            loss = self.loss_fn(score, labels)
            outputs = (loss,) + outputs
        return outputs

    
def dpr_init_from_bert(cls, model_name_or_path):
    config = BertConfig.from_pretrained(model_name_or_path)
    config._name_or_path = model_name_or_path
    config.projection_dim = 0
    model = cls(config)
    model.base_model.bert_model = BertModel.from_pretrained(model_name_or_path)
    return model


def save_models_as_hf(model,
                      tokenizer,
                      output_dir,
                      task,
                      suffix=""):

    model.q_encoder.save_pretrained(f"{output_dir}/dpr-question_encoder-single-{task}-{suffix}-base")
    model.ctx_encoder.save_pretrained(f"{output_dir}/dpr-ctx_encoder-single-{task}-{suffix}-base")
    tokenizer.save_pretrained(f"{output_dir}/dpr-question_encoder-single-{task}-{suffix}-base")
    tokenizer.save_pretrained(f"{output_dir}/dpr-ctx_encoder-single-{task}-{suffix}-base")
