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


import json
import os
import argparse
import time
import requests
from hashlib import sha256
from time import sleep
from tqdm import tqdm


def req_closed(messages, model='gpt-4o-2024-08-06', temperature=0., base_url=None, api_key=None, max_tokens=256, **kwargs):
    t = 0
    while t < 3:
        try:
            logid = sha256(messages[0]['content'].encode()).hexdigest()
            headers = {
                'Content-Type': 'application/json',
                'X-TT-LOGID': logid,
            }
            data = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            response = requests.post(
                f'{base_url}?ak={api_key}', headers=headers, json=data, timeout=30)

            return response.json()
        except Exception as e:
            t += 1
            print(messages, response, e, flush=True)
            sleep(5)
    return None


def test_closed(messages, args, tools=None):
    try:
        response = req_closed(messages=messages, model=args.model, temperature=args.temperature, tools=tools,
                              base_url=args.base_url, api_key=args.api_key, max_tokens=args.max_tokens)
        return response['choices'][0]['message']
    except Exception as e:
        print(messages, response, e, flush=True)

    return None


def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        if data_path.endswith('.json'):
            return json.load(f)
        elif data_path.endswith('.jsonl'):
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError(f"Unsupported file format: {data_path}")


def format_messages(item):
    messages = []
    for conv in item['conversations']:
        if conv['role'] == 'user':
            messages.append({"role": "user", "content": conv['content']})
        elif conv['role'] == 'assistant':
            messages.append({"role": "assistant", "content": conv['content']})
    return messages


def save_results(items, results, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for item, result in zip(items, results):
            output_item = item.copy()

            if isinstance(result, dict) and 'content' in result:
                output_item['conversations'].append(
                    {"role": "assistant", "content": result['content']})
            else:
                output_item['conversations'].append(
                    {"role": "assistant", "content": ""})

            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name to use")
    parser.add_argument("--data_path", type=str,
                        required=True, help="Path to the data file")
    parser.add_argument("--result_save_path", type=str,
                        required=True, help="Path to save the results")
    parser.add_argument("--base_url", type=str,
                        required=True, help="Base URL for the API")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key for authentication")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float,
                        default=0.0, help="Temperature for sampling")
    parser.add_argument("--save_per_num", type=int, default=10,
                        help="Save results every N samples")
    args = parser.parse_args()

    data = load_data(args.data_path)
    print(f"Loaded {len(data)} samples from {args.data_path}")

    results = []

    for i, item in enumerate(tqdm(data)):
        messages = format_messages(item)

        result = test_closed(messages, args)
        results.append(result)

        if (i + 1) % args.save_per_num == 0 or i == len(data) - 1:
            save_results(data[:i+1], results, args.result_save_path)
            print(f"Saved results for {i+1} samples")

    print(f"Completed inference for {len(data)} samples")


if __name__ == "__main__":
    main()
