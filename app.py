import os
import json
from moa.agent import MOAgent

# Update the default configuration
default_config = {
    "main_model": "phi4:latest",
    "main_system_prompt": "You are a helpful assistant. Written text should always use British English spelling.",
    "cycles": 2,
    "layer_agent_config": {
        "layer_agent_1": {
            "system_prompt": "Written text should always use British English spelling. Think through your response step by step. {helper_response}",
            "model_name": "llama3.2:3b",
            "temperature": 0.6,
        },
        "layer_agent_2": {
            "system_prompt": "Written text should always use British English spelling. Respond with a thought and then your response to the question. {helper_response}",
            "model_name": "phi4:latest",
            "temperature": 0.5,
        },
        "layer_agent_3": {
            "system_prompt": "You are an expert programmer. Written text should always use British English spelling. Always use the latest libraries and techniques. {helper_response}",
            "model_name": "qwen2.5:7b",
            "temperature": 0.3,
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
    # Configuration parameters – these can be loaded from a config file or environment variables
    main_model = default_config["main_model"]
    main_system_prompt = default_config["main_system_prompt"]
    cycles = default_config["cycles"]
    layer_agent_config = default_config["layer_agent_config"]
    main_temperature = 0.6
    main_api_base = ""#os.getenv("OLLAMA_HOST", "http://localhost:11434")
    main_api_key = ""#"ollama"
    main_num_ctx = 2048
    main_num_batch = None

    # Initialize the MOAgent for headless inference
    moa_agent = set_moa_agent(
        main_model, main_system_prompt, cycles, layer_agent_config,
        main_temperature, main_api_base, main_api_key, main_num_ctx, main_num_batch
    )

    print("Mixture of Agents (Headless Mode)")
    print("Type your question and press Enter (type 'quit' to exit).")
    while True:
        query = input(">> ")
        if query.lower() == 'quit':
            break

        # Call the chat method and accumulate the response
        response_messages = moa_agent.chat(query, output_format="json")
        response_text, final_answer = stream_response(response_messages)
        print("Response:", response_text)

        print("\nFinal Answer Begins:\n", final_answer, "\nFinal Answer Ends")

if __name__ == "__main__":
    main()