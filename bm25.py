import os
import json
from gensim.summarization import bm25
import time

start = time.time()

job = open('data/job_information_filled.json', 'r', encoding='utf-8')
person = open('data/all_645person_info.jsonl', 'r', encoding='utf-8')

corpus = []
person_list = []
for line in person:
    line = json.loads(line)
    person_list.append(line)
    corpus.append(list('专业技能：' + ','.join(line['专业技能'])))
model = bm25.BM25(corpus)

result = {}
top = [0, 0, 0, 0, 0]
R5 = []
job = json.load(job)
index = 0
for line in job:
    index += 1
    # if index < 100: continue
    # line = json.loads(line)
    # print(line)
    job_ = line['岗位描述']
    scores = model.get_scores(job_)
    scores = {i: scores[i] for i in range(len(scores))}
    scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    ids = [item[0] for item in scores[:30]]
    result[job_] = [[job_, [person_list[i] for i in ids], line["岗位编号"]]]
    for job__ in line["岗位描述-复述"]:
        scores = model.get_scores(job_)
        scores = {i: scores[i] for i in range(len(scores))}
        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ids = [item[0] for item in scores[:30]]
        result[job_].append([job__, [person_list[i] for i in ids], line["岗位编号"]])
    # result[job_] = [[{'职位': person_list[i]["职位"], '专业技能': ','.join(person_list[i]['专业技能']), '分数': scores[i][1],
    #                   '岗位编号': person_list[i]["岗位编号"]} for i in ids], line["岗位编号"]]

    # jobs = [person_list[i]["岗位编号"] for i in ids]
    # i = line["岗位编号"]
    # c = 0
    # for i_ in jobs:
    #     if i_ == i:
    #         c += 1
    # R5.append(c / 5)
    # if i in jobs[:1]:
    #     top[0] += 1
    # if i in jobs[:2]:
    #     top[1] += 1
    # if i in jobs[:3]:
    #     top[2] += 1
    # if i in jobs[:4]:
    #     top[3] += 1
    # if i in jobs:
    #     top[4] += 1
    # if i not in jobs:
    #     print(jobs, i)


json.dump(result, open('output/military_top5_no_department.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(top)
# print(sum(R5) / len(R5))
end = time.time()
print(end-start)