import json

import numpy as np

prototypes = {}
skills = set()
person = open('data/all_645person_info.jsonl', 'r', encoding='utf-8')
person_lines = []
for line in person:
    line = json.loads(line)
    skill = line["专业技能"]
    for item in skill:
        skills.add(item)
    person_lines.append(line)
skills = list(skills)
mapping_ = {skills[i]: i for i in range(len(skills))}

jobs = open('data/job_information_filled.json', 'r', encoding='utf-8')
department = {}
for line in jobs:
    line = json.loads(line)
    department[line["岗位编号"]] = line["部门"]

prototypes = {}
for line in person_lines:
    skill = line['专业技能']
    vector = [0] * len(skills)
    for item in skill:
        vector[mapping_[item]] = 1
    if department[line['岗位编号']] not in prototypes:
        prototypes[department[line['岗位编号']]] = []
    prototypes[department[line['岗位编号']]].append(vector)

for key in prototypes.keys():
    prototypes[key] = np.array(prototypes[key])
    prototypes[key] = sum(prototypes[key][i, :] for i in range(len(prototypes[key]))) / len(prototypes[key])

out = open('output/prototypes.json', 'w', encoding='utf-8')
prototypes_sparse = {}
for key in prototypes:
    value = {}
    for i in range(len(prototypes[key])):
        if prototypes[key][i] > 0:
            value[skills[i]] = prototypes[key][i]
    prototypes_sparse[key] = value
json.dump(prototypes_sparse, out, ensure_ascii=False, indent=2)
