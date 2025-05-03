import os
import json
import pdb
import time
import re
import requests

total_time = 0
total_output_length = 0
total_success = 0
num = total_success = 0
with open("gsm8k_out_llama3.1:8b.jsonl", "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
            num += 1
            total_time += entry["time"]
            total_output_length += entry["output_length"]
            total_success += entry["result"]
        except:
            print("This line can't be solved", line)
    
    average_time = total_time / num
    average_output_length = total_output_length / num
    average_success = total_success / num
    print(f"Average time: {average_time}")
    print(f"Average output length: {average_output_length}")
    print(f"Average success: {average_success * 100}%")
        
