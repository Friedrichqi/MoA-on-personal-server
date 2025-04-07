import os
import json
import pdb
from moa.agent import MOAgent
from grade_school_math.dataset import get_examples, GSMDataset, is_correct, extract_answer
from grade_school_math import sample
import time
import re
import requests

def main():
    test_instances = get_examples("test")
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "deepseek-r1:14b",
            "prompt": "Who are you?",
            "stream": False
        }
    )
    for item in test_instances:
        question = item["question"]

        start_time = time.time()
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "deepseek-r1:14b",
                "prompt": question + '\nYou should end up your answer with your final numerical result without any further explanation.',
                "stream": False
            }
        )
        elapsed_time = time.time() - start_time

        final_answer = response.json()['response']
        numbers = re.findall(r'-?[\d,]+\.?\d*', final_answer)
        numbers = [num.replace(',', '') for num in numbers]

        result = extract_answer(item["answer"]) in numbers

        updated = False
        entries = []
        if os.path.exists("gsm8k_out_deepseekr1_14b.jsonl"):
            with open("gsm8k_out_deepseekr1_14b.jsonl", "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("question") == question:
                        entry.update({
                            "golden_answer": item["answer"],
                            "chat_answer": final_answer,
                            "result": result,
                            "output_length": len(final_answer.split()),
                            "time": elapsed_time
                        })
                        updated = True
                    entries.append(entry)

        if not updated:
            entries.append({
                "question": question,
                "golden_answer": item["answer"],
                "chat_answer": final_answer,
                "result": result,
                "output_length": len(final_answer.split()),
                "time": elapsed_time
            })

        with open("gsm8k_out_deepseekr1_14b.jsonl", "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        #pdb.set_trace()
            

if __name__ == "__main__":
    main()