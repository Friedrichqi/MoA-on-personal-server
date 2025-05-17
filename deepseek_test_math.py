import os
import json
import pdb
from moa.agent import MOAgent
from MATH.dataset import get_examples, GSMDataset, is_correct, extract_answer
from MATH import sample
import time
import re
import requests
import signal

def main():
    test_instances = get_examples("test_all")
    # response = requests.post(
    #     'http://localhost:11434/api/generate',
    #     json={
    #         "model": "deepseek-r1:14b",
    #         "prompt": "Who are you?",
    #         "stream": False
    #     }
    # )
    for item in test_instances:
        problem = item["problem"]

        existed = False
        if os.path.exists("MATH_out_single_deepseek.jsonl"):
            with open("MATH_out_single_deepseek.jsonl", "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry["problem"] == problem:
                            existed = True
                            break
                    except:
                        continue
        
        if existed:
            continue
        
        def timeout_handler(signum, frame):
            raise TimeoutError()

        start_time = time.time()
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(900)
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "deepseek-r1:14b",
                    "prompt": problem + '\nYou are a mathematician expert and should end up your answer with your final numerical result without any further explanation.',
                    "stream": False
                }
            )
        except TimeoutError:
            elapsed_time = time.time() - start_time
            print("Timeout reached for this problem, skipping...")
            break  # Move on to the next test instance
        finally:
            signal.alarm(0)  # Disable the alarm

        elapsed_time = time.time() - start_time

        final_answer = response.json()['response']
        # numbers = re.findall(r'-?[\d,]+\.?\d*', final_answer)
        # numbers = [num.replace(',', '') for num in numbers]

        # result = extract_answer(item["answer"]) in numbers

        with open("MATH_out_single_deepseek.jsonl", "a") as f:
            entry = {
                "problem": item["problem"],
                "golden_answer": item["solution"],
                "extra_info": item["extra_info"],
                "chat_answer": final_answer,
                "output_length": len(final_answer.split()),
                "time": elapsed_time
            }
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()