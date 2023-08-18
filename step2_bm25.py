import os
import json
from gensim.summarization import bm25

def segmentation(string):
    tokens = list(string)
    ret = tokens.copy()
    ret.extend([i+j for i, j in zip(tokens[:-1], tokens[1:])])
    ret.extend([i+j+k for i, j, k in zip(tokens[:-2], tokens[1:-1], tokens[2:])])
    return ret

def load_corpus(path='data/all_645person_info.jsonl'):
    person = open(path, 'r', encoding='utf-8')
    corpus = []
    person_list = []
    for line in person:
        line = json.loads(line)
        person_list.append(line)
        skill = ','.join(line['专业技能'])
        skill_tokens = segmentation(skill)
        corpus.append(skill_tokens)
    model = bm25.BM25(corpus)
    return model, person_list


def query(model, job, person_list, max_count=30):
    job_descriprtion = job['岗位描述']
    job_descriprtion_tokens = segmentation(job_descriprtion)
    scores = model.get_scores(job_descriprtion_tokens)
    scores = {i: scores[i] for i in range(len(scores))}
    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = {}
    ids = [item[0] for item in scores[:max_count]]
    result[job_descriprtion] = [person_list[i] for i in ids]
    for job_ in job["岗位描述-复述"]:
        job_descriprtion_tokens = segmentation(job_)
        scores = model.get_scores(job_descriprtion_tokens)
        scores = {i: scores[i] for i in range(len(scores))}
        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ids = [item[0] for item in scores[:max_count]]
        result[job_] = [person_list[i] for i in ids]
    result['岗位编号'] = job['岗位编号']
    return result
