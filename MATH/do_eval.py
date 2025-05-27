# Licensed under the MIT license.

import sys

sys.path.append(".")

import os
import pdb
import json
from common.utils import read_json, save_json
from Evaluator import *

import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from argparse import ArgumentParser


def extract_trace(data_item, num_votes):
    res = []
    for item in data_item:
        i = 0
        trace = item["trace"] if "trace" in item else item
        rollout_id = item["rollout_id"] if "rollout_id" in item else 0
        if num_votes != -1 and rollout_id >= num_votes:
            continue
        while str(i) in trace:
            i += 1
        if "direct_answer" in trace[str(i-1)]:
            res.append(trace[str(i-1)]["direct_answer"]["text"])
        elif trace[str(i-1)]["ost_step"] != {}:
            j = 1
            while str(j) in trace[str(i-1)]["ost_step"]:
                j += 1
            res.append(trace[str(i-1)]["ost_step"][str(j-1)])
        elif "subanswer" in trace[str(i-1)]:
            res.append(trace[str(i-1)]["subanswer"]["text"])
        else:
            import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    return res


def extract_completions(data_item):
    res = []
    for item in data_item:
        res.append(data_item[item]["model_solution"])
    return res


def eval_single_item_from_answer_sheets(rollout_answer_item, evaluator: Evaluator):
    data_item = {}
    gold_answer = rollout_answer_item['extra_info']['answer']
    data_1_2 = rollout_answer_item['chat_answer']
    # model_answer_1_2, _, _, _ = evaluator.find_most_confident_answer([data_1_2])
    model_answer_1_2 = data_1_2
    result_1_2 = evaluator.check_answers_equiv(model_answer_1_2, gold_answer)
    # if not result_1_2:
    #     print(rollout_answer_item)
    #     import pdb; pdb.set_trace()
    data_item["correct"] = result_1_2
    data_item["predict_answer"] = model_answer_1_2
    data_item["gold_answer"] = gold_answer

    return data_item


def eval_exp(exp_dir: str, dataset_name: str):
    rollout_answer = []
    with open(exp_dir, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                rollout_answer.append(entry)
            except:
                # print("This line can't be solved", line)
                pass
        
    evaluator = eval(f"{dataset_name}Evaluator()")

    data_list = []
    for rollout_answer_item in tqdm(rollout_answer):
        dta = eval_single_item_from_answer_sheets(rollout_answer_item, evaluator)
        data_list.append(dta)

    # Calculate accuracy
    accuracy = sum([item["correct"] for item in data_list]) / len(data_list)
    print(f"accuracy: {accuracy}")

    # Save eval results
    answer_sheets_dir = os.path.dirname(exp_dir)
    eval_result_dir = os.path.join(answer_sheets_dir, f"{dataset_name}")
    os.makedirs(eval_result_dir, exist_ok=True)
    save_json(data_list, os.path.join(eval_result_dir, "eval_results.json"))
    analysis = {
        "accuracy": accuracy, 
        "num_tested": len(data_list),
        "num_correct": accuracy * len(data_list),
    }
    save_json(analysis, os.path.join(eval_result_dir, "analysis.json"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--exp_dir_path", type=str, required=True)
    parser.add_argument("--num_votes", type=int, default=-1)
    args = parser.parse_args()

    eval_exp(args.exp_dir_path, args.dataset_name)
