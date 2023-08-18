from transformers import BertTokenizer
from dataloader import *
from model import *
import torch
import json

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RankNet(device)
    model.load_state_dict(torch.load('best_model.pth'), False)
    model.eval()
    return model

def serialization(d):
    ret = '专业技能：' + '，'.join(d['专业技能']) + "；培训情况："
    for item in d['培训情况']:
        ret = ret + item["培训名称"] + "，" + item["培训结果"]
    ret = ret[:-1] + '；获奖情况：'
    for item in d['获奖情况']:
        ret += item['奖项名称'] + '，'
    ret = ret[:-1]
    return ret

def predict(model, tokenizer, task, candidates, max_num=10):
    scores = {}
    for i, item in enumerate(candidates):
        text = serialization(item)
        encode_dict = tokenizer.encode_plus(task, text,
                                             max_length=256,
                                             truncation=True,
                                             add_special_tokens=True,
                                             pad_to_max_length=True,
                                             return_token_type_ids=True,
                                             return_attention_mask=True)
        input_ids = torch.tensor(encode_dict['input_ids']).long().unsqueeze(0)
        input_mask = torch.tensor(encode_dict['attention_mask']).float().unsqueeze(0)
        segment_ids = torch.tensor(encode_dict['token_type_ids']).long().unsqueeze(0)
        score = model.cal_score(input_ids, input_mask, segment_ids).detach().cpu().numpy()
        scores[i] = score
    scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    return [(candidates[item[0]], item[1]) for item in scores[:max_num]]


def query(model, tokenizer, job, max_count=10):
    scores = {}
    num = -1
    if '岗位编号' in job:
        num = job['岗位编号']
        del job['岗位编号']
    for key in job.keys():
        candidates = job[key]
        filter_candidates = predict(model, tokenizer, key, candidates, max_count)
        scores[key] = {}
        for item in filter_candidates:
            scores[key][json.dumps(item[0], ensure_ascii=False)] = item[1]
    return scores, num

