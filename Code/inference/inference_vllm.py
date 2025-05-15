'''
Copyright Junjie Ye

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''


import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from typing import List, Union, Dict
import sys
import copy
from vllm import LLM, SamplingParams
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # NOQA
from format_for_vllm import *


QWEN1_5_TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"
QWEN1_TEMPLATE = QWEN1_5_TEMPLATE
QWEN2_TEMPLATE_OFFICE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
QWEN2_TEMPLATE_OURS = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
LLAMA3_1_TEMPLATE = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
LLAMA2_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
GEMMA2_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
GEMMA_TEMPLATE = GEMMA2_TEMPLATE


def data2jsonl_file(data: Union[List[Dict], Dict], file_name, mode="w"):
    with open(file_name, mode=mode, encoding="utf-8") as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def inference_vllm(args):

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", trust_remote_code=True)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    stop_token_ids = []
    if args.model_type in ["auto", "qwen2", "llama3.1"]:
        if args.model_type == "auto":
            pass
        elif args.model_type == "llama3.1":
            tokenizer.pad_token = '<|end_of_text|>'
            if tokenizer.chat_template == None:
                tokenizer.chat_template = LLAMA3_1_TEMPLATE
            # 128001:<|end_of_text|> 128008:<|eom_id|> 128009:<|eot_id|>
            stop_token_ids = [128001, 128008, 128009]
        elif args.model_type == "qwen2":
            tokenizer.pad_token = '<|endoftext|>'
            tokenizer.pad_token_id = 151643
            tokenizer.chat_template = QWEN2_TEMPLATE_OURS
            # 151645:<|im_end|> 151643:<|endoftext|>
            stop_token_ids = [151645, 151643]
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True
    )
    llm.set_tokenizer(tokenizer)

    sampling_params = SamplingParams(
        stop_token_ids=stop_token_ids,
        max_tokens=args.max_new_tokens,
        n=args.sampling_times,
        temperature=args.temperature,
    )

    data = format_test_data(data_path=args.data_path)

    output_datas = []
    for bp in tqdm(range(0, len(data), args.batch_size), desc="Batch Inference of data"):
        split_data = data[bp:bp + args.batch_size] if (
            bp + args.batch_size) < len(data) else data[bp:]

        texts = []
        for d in split_data:
            messages = d["conversations"]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        outputs = llm.generate(texts, sampling_params)

        print(
            f"[Prompt]{'=='*30}\n{outputs[0].prompt}\n[Generated text]{'=='*30}\n{outputs[0].outputs[0].text}{'=='*10}")

        for i, output in enumerate(outputs):
            output_data = split_data[i]
            for j, res in enumerate(output.outputs):
                output_new = copy.deepcopy(output_data)
                time_suffix = f"_times_{j}/{args.sampling_times}" if args.sampling_times > 1 else ""
                output_new["id"] = str(output_new["id"]) + time_suffix
                output_new["conversations"].append(
                    {"role": "assistant", "content": res.text.strip()})
                output_datas.append(output_new)

        if args.save_per_num is not None and len(output_datas) >= args.save_per_num:
            data2jsonl_file(data=output_datas,
                            file_name=args.result_save_path, mode='a')
            output_datas = []

    if len(output_datas) != 0:
        data2jsonl_file(data=output_datas,
                        file_name=args.result_save_path, mode='a')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default=None, help="Model Checkpoint path"
    )
    parser.add_argument(
        "--model_type", type=str, default="auto", help="Model Type, eg: llama3.1, llama2, qwen2"
    )
    parser.add_argument(
        "--data_path",  type=str, help="Inference Data Path"
    )
    parser.add_argument(
        "--result_save_path", type=str, default=None, help="result save path (jsonl)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Inference batch size"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048, help="max_new_tokens"
    )
    parser.add_argument(
        "--save_per_num", type=int, default=32, help="The number of data intervals saved"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature"
    )
    parser.add_argument(
        "--sampling_times", type=int, default=1, help="The number of samples for each data"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="The number of GPUs to use for distributed execution with tensor parallelism"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9, help="The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache. Higher values will increase the KV cache size and thus improve the model's  throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors."
    )

    args = parser.parse_args()

    inference_vllm(args)
