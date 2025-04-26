import os
import json
import pdb
from moa.agent import MOAgent
from grade_school_math.dataset import get_examples, GSMDataset, is_correct, extract_answer
from grade_school_math import sample
import time
import re

# Update the default configuration
default_config = {
    "main_model": "deepseek-r1:14b",
    "main_system_prompt": "You are a mathematician expert. You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. End up your answer with a single numerical value without further explanation.",
    "main_temperature": 0.6,
    "main_api_base": "",
    "main_api_key": "",
    "main_num_ctx": 2048,
    "main_num_batch": None,
    "cycles": 2,
    "layer_agent_config": {
        "layer_agent_1": {
            "system_prompt": "Written text should always use British English spelling. Think through your response step by step. {helper_response}",
            "model_name": "qwen2-math:7b",
            "temperature": 0.75,
        },
        "layer_agent_2": {
            "system_prompt": "Written text should always use British English spelling. Respond with a thought and then your response to the question. {helper_response}",
            "model_name": "qwen2-math:7b",
            "temperature": 0.5,
        },
        "layer_agent_3": {
            "system_prompt": "You are a mathematician expert. Written text should always use British English spelling. Always use the latest libraries and techniques. {helper_response}",
            "model_name": "qwen2-math:7b",
            "temperature": 0.25,
        },
    },
}

def stream_response(messages):
    """Accumulate the response from a series of message chunks."""
    response = final_answer = ""
    for message in messages:
        if message['response_type'] != 'output':
            response += '\n' + str(message['metadata'].get('layer')) + ':\n' + message["delta"]
        else:
            final_answer += message["delta"]
    return response, final_answer

def set_moa_agent(main_model, main_system_prompt, cycles, layer_agent_config,
                  main_temperature, main_api_base, main_api_key, main_num_ctx, main_num_batch, **optional_params):
    """Initialize MOAgent with the given configuration."""
    main_model_kwargs = {"temperature": main_temperature}
    # Add optional parameters if provided
    for key, value in optional_params.items():
        if value is not None and value != 0:
            main_model_kwargs[key] = value

    if main_api_base:
        main_model_kwargs["api_base"] = main_api_base
    if main_api_key:
        main_model_kwargs["api_key"] = main_api_key
    if main_num_ctx and main_num_ctx > 0:
        main_model_kwargs["num_ctx"] = main_num_ctx
    if main_num_batch and main_num_batch > 0:
        main_model_kwargs["num_batch"] = main_num_batch

    return MOAgent.from_config(
        main_model=main_model,
        main_system_prompt=main_system_prompt,
        cycles=cycles,
        layer_agent_config=layer_agent_config,
        **main_model_kwargs,
    )

def main():
    main_model = default_config["main_model"]
    main_system_prompt = default_config["main_system_prompt"]
    main_temperature = default_config["main_temperature"]
    main_api_base = default_config["main_api_base"]
    main_api_key = default_config["main_api_key"]
    main_num_ctx = default_config["main_num_ctx"]
    main_num_batch = default_config["main_num_batch"]
    cycles = default_config["cycles"]
    layer_agent_config = default_config["layer_agent_config"]

    # Initialize the MOAgent for headless inference
    moa_agent = set_moa_agent(
        main_model, main_system_prompt, cycles, layer_agent_config,
        main_temperature, main_api_base, main_api_key, main_num_ctx, main_num_batch
    )

    # Initial Start-up
    test_instances = get_examples("test")
    moa_agent.chat("how are you?", output_format="json")

    for item in test_instances:
        question = item["question"]

        existed = False
        if os.path.exists("gsm8k_out.jsonl"):
            with open("gsm8k_out.jsonl", "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry["question"] == question:
                            existed = True
                            break
                    except:
                        # print("Error processing line:", line)
                        continue
        
        
        # Skip finished questions
        if existed:
            continue
        
        start_time = time.time()
        response_messages = list(moa_agent.chat(question, output_format="json"))
        elapsed_time = time.time() - start_time

        response_text , final_answer = stream_response(response_messages)
        numbers = re.findall(r'-?[\d,]+\.?\d*', final_answer)
        numbers = [num.replace(',', '') for num in numbers]

        # result = is_correct(final_numerical_value, item)
        result = extract_answer(item["answer"]) in numbers

        with open("gsm8k_out.jsonl", "a") as f:
            entry = {
                "question": question,
                "golden_answer": item["answer"],
                "chat_answer": final_answer,
                "result": result,
                "output_length": len(final_answer.split()),
                "time": elapsed_time
            }
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()