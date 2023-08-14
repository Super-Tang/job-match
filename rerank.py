from transformers import BertTokenizer
from dataloader import *
from model import *
import torch
import json

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RankNet(device)
    model.load_state_dict(torch.load('best_model1.pth'), False)
    model.eval()
    return model

def serialization(d):
    ret = "职位：" + d['职位'] + '；专业技能：' + '，'.join(d['专业技能']) + "；培训情况："
    for item in d['培训情况']:
        ret = ret + item["培训名称"] + "，" + item["培训结果"]
    ret = ret + '；工作时间：' + d['工作时间'] + '；获奖情况：'
    for item in d['获奖情况']:
        ret += item['奖项名称'] + '，'
    ret = ret[:-1] + "；健康情况：" + str(d['健康情况'])
    return ret

def predict(model, tokenizer, task, candidates, max_num=5):
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
    return [(candidates[item[0]], item[1]) for item in scores[:10]]


def test(model, tokenizer, file_path):
    data = json.load(open(file_path, 'r', encoding='utf-8'))
    filter_data = {}
    filter_data_all = {}
    for key in data.keys():
        temp_scores = {}
        for example in data[key]:
            task = example[0]
            candidates = example[1]
            filter_candidate = predict(model, tokenizer, key, candidates)
            for item in filter_candidate:
                if json.dumps(item[0], ensure_ascii=False) not in temp_scores:
                    temp_scores[json.dumps(item[0], ensure_ascii=False)] = item[1]/4
                else:
                    temp_scores[json.dumps(item[0], ensure_ascii=False)] += item[1]/4
        temp_scores = sorted(temp_scores.items(), key=lambda x:x[1], reverse=True)
        filter_candidate = [(json.loads(item[0]), item[1]) for item in temp_scores[:5]]
        filter_data[key] = []
        filter_data_all[key] = [[candidate[0] for candidate in filter_candidate], data[key][0][2]]
        for item in filter_candidate:
            info = f"姓名：{item[0]['姓名']}，职位：{item[0]['职位']}，工作科室：{item[0]['工作部门']}，专业技能：{item[0]['专业技能']}，得分：{item[1]}，岗位编号：[{item[0]['岗位编号']},{data[key][0][2]}]"
            filter_data[key].append(info)
    json.dump(filter_data, open('result.json', 'w', encoding='utf-8'),  ensure_ascii=False, indent=2)
    json.dump(filter_data_all, open('result_all.json', 'w', encoding='utf-8'),  ensure_ascii=False, indent=2)



model = load_model()
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
test(model, tokenizer, 'military.json')