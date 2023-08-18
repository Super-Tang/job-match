import json

data = json.load(open('result.json', 'r', encoding='utf-8'))
top = [[0, 0, 0, 0, 0] for _ in range(5)]
R5 = [[] for _ in range(5)]
for _, key in enumerate(data.keys()):
    value = data[key]
    for key_ in value.keys():
        if '-' in key_:
            index = int(key_.split('-')[1])
            value_ = value[key_]
            jobs = [int(value__.split('：')[-1].split(',')[0][1:]) for value__ in value_]
            i = int(value_[0].split('：')[-1].split(',')[1][:-1])
            c = 0
            for i_ in jobs:
                if i == i_: c += 1
            R5[index].append(c / 5)
            if i in jobs[:1]:
                top[index][0] += 1
            if i in jobs[:2]:
                top[index][1] += 1
            if i in jobs[:3]:
                top[index][2] += 1
            if i in jobs[:4]:
                top[index][3] += 1
            if i in jobs:
                top[index][4] += 1
            if c == 0:
                print(jobs, i)
        else:
            index = 4
            value_ = value[key_]
            jobs = [int(value__.split('：')[-1].split(',')[0][1:]) for value__ in value_]
            i = int(value_[0].split('：')[-1].split(',')[1][:-1])
            c = 0
            for i_ in jobs:
                if i == i_: c += 1
            R5[index].append(c / 5)
            if i in jobs[:1]:
                top[index][0] += 1
            if i in jobs[:2]:
                top[index][1] += 1
            if i in jobs[:3]:
                top[index][2] += 1
            if i in jobs[:4]:
                top[index][3] += 1
            if i in jobs:
                top[index][4] += 1
            if c == 0:
                print(jobs, i)
print(top, len(data.keys()))
print([sum(R5_) / len(R5_) for R5_ in R5])
