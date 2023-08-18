import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics as mr
import pandas as pd

job = open('data/310_job_info(2).json', 'r', encoding='utf-8')
person = open('data/all_1550person_info(1).json', 'r', encoding='utf-8')

job_dict = {}
corpus = []
for line in job:
    line = json.loads(line)
    job_dict[line["岗位编号"]] = line["职责内容"]
    corpus.append(list(line['职责内容']))

person_dict = {"other_info":[], "reward":[], "skill":[], "training":[], 'task':[]}

def serialization(d):
    ret = []
    for item in d:
        for key in item.keys():
            ret.append(item[key])
    return ','.join(ret)


for line in person:
    line = json.loads(line)
    other_info = list(serialization(line['其他相关信息']))
    reward_info = list(serialization(line['获奖情况']))
    skills = list(','.join(line['专业技能']))
    training = list(serialization(line['培训情况']))
    task = job_dict[int(line["岗位编号"])]
    person_dict['other_info'].append(other_info)
    person_dict['reward'].append(reward_info)
    person_dict['skill'].append(skills)
    person_dict['training'].append(training)
    person_dict['task'].append(task)
    corpus.extend([other_info, reward_info, skills, training])

corpus = [' '.join(item) for item in corpus]
verctorizer = TfidfVectorizer(analyzer='char')
person_df = pd.DataFrame().from_dict(person_dict)

tfidf = verctorizer.fit_transform(corpus)

def sim(a, b):
    return  sum(a*b) / sum(a**2)**0.5 / sum(b**2)**0.5

scores = {}
for _, row in person_df.iterrows():
    other_info = verctorizer.transform([' '.join(list(row['other_info']))]).toarray()
    task = verctorizer.transform([' '.join(list(row['task']))]).toarray()
    reward = verctorizer.transform([' '.join(list(row['reward']))]).toarray()
    skill = verctorizer.transform([' '.join(list(row['skill']))]).toarray()
    training = verctorizer.transform([' '.join(list(row['training']))]).toarray()
    # print(mr.mutual_info_score(task[0], training[0]))
    # print(mr.mutual_info_score(task[0], skill[0]))
    # print(mr.mutual_info_score(task[0], reward[0]))
    # print(mr.mutual_info_score(task[0], other_info[0]))
    if 'task_reward' not in scores:
        scores['task_reward'] = []
        scores['task_skill'] = []
        scores['task_training'] = []
        scores['task_other'] = []
        scores['reward_skill'] = []
        scores['reward_training'] = []
        scores['reward_other'] = []
        scores['skill_training'] = []
        scores['skill_other'] = []
        scores['training_other'] = []
    scores['task_reward'].append(mr.adjusted_mutual_info_score(task[0], reward[0]))
    scores['task_skill'].append(mr.adjusted_mutual_info_score(task[0], skill[0]))
    scores['task_training'].append(mr.adjusted_mutual_info_score(task[0], training[0]))
    scores['task_other'].append(mr.adjusted_mutual_info_score(task[0], other_info[0]))
    scores['reward_skill'].append(mr.adjusted_mutual_info_score(reward[0], skill[0]))
    scores['reward_training'].append(mr.adjusted_mutual_info_score(reward[0], training[0]))
    scores['reward_other'].append(mr.adjusted_mutual_info_score(reward[0], other_info[0]))
    scores['skill_other'].append(mr.adjusted_mutual_info_score(skill[0], other_info[0]))
    scores['skill_training'].append(mr.adjusted_mutual_info_score(skill[0], training[0]))
    scores['training_other'].append(mr.adjusted_mutual_info_score(training[0], other_info[0]))

from pprint import pprint
pprint([(key, sum(scores[key])/len(scores[key])) for key in scores.keys()])





