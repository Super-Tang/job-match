import json
import random
from itertools import product
random.seed(761)

person = open('data/all_645person_info.jsonl', 'r', encoding='utf-8')
job = json.load(open('data/job_information_all.json', 'r', encoding='utf-8'))

job_info = {}
for line in job:
    if line["岗位编号"] not in job_info:
        job_info[line["岗位编号"]] = []
    job_info[line["岗位编号"]].append([line["岗位"] + ':' + line['岗位描述'], line["排序"]])

person_info = {}
for line in person:
    line = json.loads(line)
    if int(line['岗位编号']) not in person_info:
        person_info[int(line['岗位编号'])] = []
    person_info[int(line['岗位编号'])].append(line)


def serialiation(d):
    """
    对人的信息进行序列化，构造字符串，使用的信息包括，职位，专业技能，培训情况，工作时间等
    :param d: 字典，候选人的相关信息
    :return: 字符串
    """
    ret = '专业技能：' + '，'.join(d['专业技能']) + "；培训情况："
    for item in d['培训情况']:
        ret = ret + item["培训名称"] + "，" + item["培训结果"] + ','
    ret = ret[:-1] + '；获奖情况：'
    for item in d['获奖情况']:
        ret += item['奖项名称'] + '，'
    ret = ret[:-1]
    return ret


data = []
all_samples = [serialiation(sample) for choice in job_info.keys() for sample in person_info[choice]]

score_mapping = {
    '优秀': 10,
    '良好': 7
}

def rerank(candidates):
    def cal_score(candidate):
        score = 0
        score += candidate['健康情况']
        for item in candidate['培训情况']:
            score += score_mapping[item['培训结果']]
        score += 5 * len(candidate['获奖情况'])
        score += int(candidate['工作时间'][:-1])
        return score

    scores = [0, 0, 0, 0, 0]
    for i in range(len(candidates)):
        scores[i] = cal_score(candidates[i])
    return scores

positive = 0
count = 0
test_task = {}
choices = [i for i in range(len(job_info.keys()))]
random.shuffle(choices)
selected = choices[:100]
print(len(selected))
print(job_info)
for key in job_info:
    if count not in selected:
        count += 1
        test_task[key] = job_info[key]
        continue
    count += 1
    for item in job_info[key]:
        job_description, _ = item
        personnel_candidates = person_info[key]
        # rank = rerank(personnel_candidates)
        # tuples = list(product(range(len(personnel_candidates)), range(len(personnel_candidates))))
        # for i, j in tuples:
        #     if i != j:
        #         item1 = serialiation(personnel_candidates[i])
        #         item2 = serialiation(personnel_candidates[j])
        #         rank1, rank2 = rank[i], rank[j]
        #         if rank1 < rank2:
        #             data.append({'job_description': job_description, 'item1': item1, 'item2': item2, 'label': 0})
        #         else:
        #             data.append({'job_description': job_description, 'item1': item1, 'item2': item2, 'label': 1})
        #             positive += 1
        for personnel_candidate in personnel_candidates:
            info = serialiation(personnel_candidate)
            negative_samples = random.sample(all_samples, 20)
            data.extend([{'job_description': job_description, 'item1': info, 'item2': negative_sample, 'label': 1} for
                         negative_sample in negative_samples[:10] if negative_sample != info])
            positive += 5
            data.extend([{'job_description': job_description, 'item2': info, 'item1': negative_sample, 'label': 0} for
                         negative_sample in negative_samples[10:] if negative_sample != info])
            negative_samples1 = random.sample(all_samples, 5)
            negative_samples2 = random.sample(all_samples, 5)
            for i, j in zip(negative_samples1, negative_samples2):
                if i != j and info not in [i, j]:
                    data.append({'job_description': job_description, 'item2': i, 'item1': j, 'label': 0})

out = open('data_761.json', 'w', encoding='utf-8')
test = open('test_761.json', 'w', encoding='utf-8')
print(len(data), positive)
json.dump(data, out, ensure_ascii=False, indent=2)
json.dump(test_task, test, ensure_ascii=False, indent=2)
