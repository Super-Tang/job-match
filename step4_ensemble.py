import json
import math


def cal_score(candidate):
    score_mapping = {
        '优秀': 10,
        '良好': 7
    }
    candidate = json.loads(candidate)
    score = 0
    score += candidate['健康情况']
    for item in candidate['培训情况']:
        score += score_mapping[item['培训结果']]
    score += 5 * len(candidate['获奖情况'])
    score += int(candidate['工作时间'][:-1])
    return score

def sigmoid(x):
    return 2*x/100

def ensemble(result, max_count=5):
    ret = {}
    all_result = {}
    for i, key in enumerate(result.keys()):
        all_result['paraphrase-'+str(i)] = []
        scores = result[key]
        for index, item in enumerate(scores.keys()):
            if item not in ret:
                ret[item] = 0
            elif i == 0: ret[item] = scores[item] * (10-index)
            if i > 0:
                ret[item] += scores[item] / (2+2*i) * (10-index)
            if len(all_result['paraphrase-' + str(i)]) < 5: all_result['paraphrase-' + str(i)].append((item, scores[item]))
    for item in ret:
        ret[item] = ret[item] * sigmoid(cal_score(item))
    ret = sorted(ret.items(), key=lambda x:x[1], reverse=True)
    all_result['final'] = ret[:max_count]
    return all_result
