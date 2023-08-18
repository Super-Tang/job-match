import json
import time
import step2_bm25
import step3_rank
import step4_ensemble
from transformers import BertTokenizer
MAX_BM25=30
MAX_RANK=10
MAX_ENSEMBLE=5


def pipeline(bm25_model, rank_model, person_list, tokenizer, job):
    bm25_result = step2_bm25.query(bm25_model, job, person_list, max_count=MAX_BM25)
    rank_result, num = step3_rank.query(rank_model, tokenizer, bm25_result, max_count=MAX_RANK)
    final_result = step4_ensemble.ensemble(rank_result, max_count=MAX_ENSEMBLE)
    return final_result, num


def main():
    bm25_model, person_list = step2_bm25.load_corpus(path='all_645person_info.jsonl')
    rank_model = step3_rank.load_model()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    jobs = json.load(open('job_information_augmented.json', 'r', encoding='utf-8'))
    result_simple = {}
    result_all = {}
    for job in jobs:
        result, num = pipeline(bm25_model, rank_model, person_list, tokenizer, job)
        job_desc = job['岗位描述']
        result_simple[job_desc] = {}
        for key in result:
            result_simple[job_desc][key] = []
            for candidate, score in result[key]:
                item = json.loads(candidate)
                info = [f"姓名：{item['姓名']}，职位：{item['职位']}，工作科室：{item['工作部门']}，专业技能：{item['专业技能']}，得分：{score}，岗位编号：[{item['岗位编号']},{num}]", score]
                result_simple[job_desc][key].append(info)
    json.dump(result_simple, open('result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

start = time.time()
main()
end = time.time()
print(end-start)