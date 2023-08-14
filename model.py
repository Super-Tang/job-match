import torch.nn.functional as F
from torch import nn
from transformers import BertModel


class RankNet(nn.Module):
    def __init__(self, device, dropout=0.1):
        super(RankNet, self).__init__()
        self.device = device
        self.bert = BertModel.from_pretrained("bert-base-chinese").to(self.device)
        self.dropout = nn.Dropout(dropout)
        self.shared_nn = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 64).to(self.device),
            nn.ReLU()
        ).to(self.device)
        self.score_diff = nn.Sequential(
            nn.Linear(64, 1).to(self.device),
            nn.Sigmoid()
        ).to(self.device)


    def forward(self, input_ids0, input_mask0, segment_ids0, input_ids1, input_mask1, segment_ids1, label_id=None):
        input_ids0 = input_ids0.to(self.device)
        input_mask0 = input_mask0.to(self.device)
        segment_ids0 = segment_ids0.to(self.device)
        bert_output0 = self.bert(input_ids=input_ids0, attention_mask=input_mask0, token_type_ids=segment_ids0)
        cls0 = bert_output0[1]

        input_ids1 = input_ids1.to(self.device)
        input_mask1 = input_mask1.to(self.device)
        segment_ids1 = segment_ids1.to(self.device)
        bert_output1 = self.bert(input_ids=input_ids1, attention_mask=input_mask1, token_type_ids=segment_ids1)
        cls1 = bert_output1[1]

        shared_nn_a = self.shared_nn(cls0)
        shared_nn_b = self.shared_nn(cls1)
        score_diff = self.score_diff(shared_nn_a - shared_nn_b)
        return score_diff

    def cal_score(self, input_ids, input_mask, segment_ids):
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)
        bert_output = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        cls = bert_output[1]
        shared_nn = self.shared_nn(cls)
        score = self.score_diff(shared_nn)
        return score


def focal_loss(pred, target, gamma=2, alpha=0.25):
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target) * focal_weight
    return loss.mean()