import json
from random import shuffle

f = json.load(open('data/job_information_augmented.json', 'r', encoding='utf-8'))
out = []
for line in f:
    nums = [0, 1, 2, 3, 4]
    common = {
        "部门": line["部门"],
        "岗位": line["岗位"],
        "岗位编号": line["岗位编号"]
    }
    original = common.copy()
    original["岗位描述"] = line["岗位描述"]
    shuffle(nums)
    original['排序'] = nums
    out.append(original)
    for paraphrase in line['岗位描述-复述']:
        nums = [0, 1, 2, 3, 4]
        augmented = common.copy()
        augmented["岗位描述"] = paraphrase
        shuffle(nums)
        augmented['排序'] = nums
        out.append(augmented)
json.dump(out, open('data/job_information_all.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)