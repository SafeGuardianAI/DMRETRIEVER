import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from transformers.file_utils import ModelOutput

import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    def __init__(
        self,
        backbone,
        tokenizer=None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        distill_loss_weight: float = 1.0,
        sentence_pooling_method: str = 'cls',
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer

        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.sub_batch_size = sub_batch_size
        self.kd_loss_type = kd_loss_type
        self.distill_loss_weight = distill_loss_weight
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self._keys_to_ignore_on_save = []
        self._keys_to_ignore_on_load_missing = []
        self._keys_to_ignore_on_load_unexpected = []

    def encode(self, features):
        if features is None:
            return None
        if not isinstance(features, list):
            if self.sub_batch_size is not None and self.sub_batch_size > 0:
                all_reps = []
                for i in range(0, len(features['attention_mask']), self.sub_batch_size):
                    end = min(i + self.sub_batch_size, len(features['attention_mask']))
                    sub_features = {k: v[i:end] for k, v in features.items()}
                    hidden = self.backbone(**sub_features, return_dict=True).last_hidden_state
                    reps = self._pool(hidden, sub_features['attention_mask'])
                    all_reps.append(reps)
                all_reps = torch.cat(all_reps, 0).contiguous()
                if self.normalize_embeddings:
                    all_reps = F.normalize(all_reps, dim=-1)
                return all_reps.contiguous()
            else:
                hidden = self.backbone(**features, return_dict=True).last_hidden_state
                all_reps = self._pool(hidden, features['attention_mask'])
                if self.normalize_embeddings:
                    all_reps = F.normalize(all_reps, dim=-1)
                return all_reps.contiguous()
        else:
            all_reps = []
            for sub_features in features:
                hidden = self.backbone(**sub_features, return_dict=True).last_hidden_state
                reps = self._pool(hidden, sub_features['attention_mask'])
                all_reps.append(reps)
            all_reps = torch.cat(all_reps, 0).contiguous()
            if self.normalize_embeddings:
                all_reps = F.normalize(all_reps, dim=-1)
            return all_reps.contiguous()

    def _pool(self, last_hidden_state, attention_mask):
        if self.sentence_pooling_method == "cls":
            return last_hidden_state[:, 0]
        elif self.sentence_pooling_method == "mean":
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                return last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device),
                    sequence_lengths,
                ]
        else:
            raise ValueError(f"Unknown pooling method: {self.sentence_pooling_method}")

    def _similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1)) / self.temperature
        return torch.matmul(q_reps, p_reps.transpose(-2, -1)) / self.temperature

    def forward(
        self,
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        q_reps = self.encode(queries)
        p_reps = self.encode(passages)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device)
                teacher_scores = teacher_scores.view(q_reps.size(0), -1).detach()
                teacher_targets = F.softmax(teacher_scores, dim=-1)
            else:
                teacher_targets = None

            if no_in_batch_neg_flag:
                compute_loss_func = self._loss_no_in_batch_neg
            else:
                compute_loss_func = (
                    self._loss_cross_device_neg if self.negatives_cross_device
                    else self._loss_in_batch_neg
                )

            scores, loss = compute_loss_func(q_reps, p_reps, teacher_targets=teacher_targets)
            return EncoderOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)
        else:
            scores = (
                self._similarity(q_reps, p_reps)
                if q_reps is not None and p_reps is not None
                else None
            )
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps, scores=scores)

    def _get_local_scores(self, q_reps, p_reps, all_scores):
        group_size = p_reps.size(0) // q_reps.size(0)
        indices = torch.arange(0, q_reps.size(0), device=q_reps.device) * group_size
        specific_scores = []
        for i in range(group_size):
            specific_scores.append(
                all_scores[torch.arange(q_reps.size(0), device=q_reps.device), indices + i]
            )
        return torch.stack(specific_scores, dim=1).view(q_reps.size(0), -1)

    def _loss_no_in_batch_neg(self, q_reps, p_reps, teacher_targets=None):
        group_size = p_reps.size(0) // q_reps.size(0)
        scores = self._similarity(q_reps, p_reps)
        local_scores = self._get_local_scores(q_reps, p_reps, scores)

        if teacher_targets is not None:
            if self.kd_loss_type == "kl_div":
                kd_loss = self._distill_loss(self.kd_loss_type, teacher_targets, local_scores, group_size=group_size)
                targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long)
                ce_loss = self.cross_entropy(local_scores, targets)
                loss = ce_loss + self.distill_loss_weight * kd_loss
            else:
                kd_loss = self._distill_loss(self.kd_loss_type, teacher_targets, local_scores, group_size=group_size)
                loss = self.distill_loss_weight * kd_loss
        else:
            targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long)
            loss = self.cross_entropy(local_scores, targets)

        return local_scores, loss

    def _loss_in_batch_neg(self, q_reps, p_reps, teacher_targets=None):
        group_size = p_reps.size(0) // q_reps.size(0)
        scores = self._similarity(q_reps, p_reps)

        if teacher_targets is not None:
            if self.kd_loss_type == "kl_div":
                student_scores = self._get_local_scores(q_reps, p_reps, scores)
                kd_loss = self._distill_loss(self.kd_loss_type, teacher_targets, student_scores, group_size)
                idxs = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
                targets = idxs * group_size
                ce_loss = self.cross_entropy(scores, targets)
                loss = ce_loss + self.distill_loss_weight * kd_loss
            elif self.kd_loss_type == "m3_kd_loss":
                kd_loss = self._distill_loss(self.kd_loss_type, teacher_targets, scores, group_size)
                loss = self.distill_loss_weight * kd_loss
            else:
                raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")
        else:
            idxs = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
            targets = idxs * group_size
            loss = self.cross_entropy(scores, targets)

        return scores, loss

    def _loss_cross_device_neg(self, q_reps, p_reps, teacher_targets=None):
        group_size = p_reps.size(0) // q_reps.size(0)

        cross_q_reps = self._dist_gather_tensor(q_reps)
        cross_p_reps = self._dist_gather_tensor(p_reps)
        cross_scores = self._similarity(cross_q_reps, cross_p_reps)

        if teacher_targets is not None:
            if self.kd_loss_type == "kl_div":
                student_scores = self._get_local_scores(cross_q_reps, cross_p_reps, cross_scores)
                student_scores = student_scores[
                    q_reps.size(0) * self.process_rank : q_reps.size(0) * (self.process_rank + 1)
                ]
                kd_loss = self._distill_loss(self.kd_loss_type, teacher_targets, student_scores, group_size)
                cross_idxs = torch.arange(cross_q_reps.size(0), device=cross_q_reps.device, dtype=torch.long)
                cross_targets = cross_idxs * group_size
                ce_loss = self.cross_entropy(cross_scores, cross_targets)
                loss = ce_loss + self.distill_loss_weight * kd_loss
            elif self.kd_loss_type == "m3_kd_loss":
                cross_teacher_targets = self._dist_gather_tensor(teacher_targets)
                kd_loss = self._distill_loss(self.kd_loss_type, cross_teacher_targets, cross_scores, group_size)
                loss = self.distill_loss_weight * kd_loss
            else:
                raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")
        else:
            cross_idxs = torch.arange(cross_q_reps.size(0), device=cross_q_reps.device, dtype=torch.long)
            cross_targets = cross_idxs * group_size
            loss = self.cross_entropy(cross_scores, cross_targets)

        return cross_scores, loss

    @staticmethod
    def _distill_loss(kd_loss_type, teacher_targets, student_scores, group_size=None):
        if kd_loss_type == 'kl_div':
            return -torch.mean(torch.sum(torch.log_softmax(student_scores, dim=-1) * teacher_targets, dim=-1))
        elif kd_loss_type == 'm3_kd_loss':
            labels = torch.arange(student_scores.size(0), device=student_scores.device, dtype=torch.long)
            labels = labels * group_size
            loss = 0
            mask = torch.zeros_like(student_scores)
            for i in range(group_size):
                temp_target = labels + i
                temp_scores = student_scores + mask
                temp_loss = F.cross_entropy(temp_scores, temp_target, reduction="none")
                loss += torch.mean(teacher_targets[:, i] * temp_loss)
                mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                     value=torch.finfo(student_scores.dtype).min)
            return loss
        else:
            raise ValueError(f"Invalid kd_loss_type: {kd_loss_type}")

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.backbone.enable_input_require_grads(**kwargs)

    def save(self, output_dir: str):
        state_dict = self.backbone.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.backbone.save_pretrained(output_dir, state_dict=state_dict)
