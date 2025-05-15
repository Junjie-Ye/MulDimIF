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


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # NOQA
from Code.utils.data_utils import *
import re
import argparse


templatef1 = '''You are an expert in constructing data based on instructions. You need to generate the corresponding data as required.
You should modify the given 【Original Question】 according to the 【Core Requirements】 without changing the original meaning of the question. Then, respond in the format specified in the 【Reply Format】.

【Core Requirements】:
    1. Fully understand the 【Original Question】 and the constraints listed in the 【Constraint List】.
    2. Change the expression of the 【Original Question】. First, extract the core question from the 【Original Question】 that is not bound by constraints, then list the constraints corresponding to the 【Constraint List】 at the end of the sentence. Start with "The output must follow the following rules:" and list the constraints from the 【Original Question】 clearly after understanding the constraints.
    3. The modified question must remain consistent with the 【Original Question】 in terms of meaning and constraints.

【Reply Format】: 
    【Constraint List Data】: Core question (does not include constraint descriptions in the constraint list), \nThe output must follow the following rules: 
    1.xxx 2.xxx

【Data】: 
    【Original Question】:{original_question} 
    【Constraint List】:{constraint_list}'''

templatef2 = '''You are an expert in data construction based on instructions. You need to generate the corresponding data as required.
You should modify the given 【Data】 according to the 【Core Requirements】 without changing the original meaning of the question. Then, respond in the format specified in the 【Reply Format】.

【Core Requirements】:
    1. Do not alter the question to directly satisfy the constraints.
    2. Fully understand the 【Original Question】 and the constraints within it.
    3. Modify the expression of the constraints in the 【Original Question】 by clearly describing them in the question, so that the question explicitly indicates the constraints, without changing its structure to meet those constraints directly.
    4. The modified question should keep the original meaning and intent, while the constraints are introduced as descriptive explanations or clarifications in the question.
    5. Ensure that the constraints are explicitly described in the question, making it clear that they need to be considered when answering, without altering the question to directly satisfy them.

【Reply Format】: 
    【Constraint Integration Format Data】: xxx

【Data】: 
    【Original Question】:{original_question} 
    【Constraint List】:{constraint_list}'''

templatec1 = '''You are an expert in following instructions to construct data. You need to conduct a series of checks on the given data according to the requirements.
You are to check the given【Data】according to the【Core requirements】and respond in the format specified in【Reply format】.

【Core requirements】:
    1. Ensure all listed constraints are consistent with the original problem requirements.

【Reply format】: 
    【Specific explanation】: xxx
    【Is the listed constraint form question clearly stated to cover all constraints?】: [Yes/No]

【Data】:
    【Original question】: {original_question}
    【Listed constraint form question】:{listed_constraint_form_question}
    【Constraint list】: {constraint_list}'''

templatec2 = '''You are an expert in following instructions to construct data. You need to conduct a series of checks on the given data according to the requirements.
You are to check the given【Data】according to the【Core requirements】and respond in the format specified in【Reply format】.

【Core requirements】:  
    1. Ensure that the question in the integrated constraint form is consistent with the original problem requirements.  

【Reply format】:  
    【Specific explanation】: xxx  
    【Does the question in the integrated constraint form clearly cover all constraints listed?】: [Yes/No]  

【Data】:
    【Original question】: {original_question}
    【Integrated constraint form question】:{integrated_constraint_form_question}
    【Constraint list】: {constraint_list}'''


# generate interaction prompt for constraint extension
'''
def generate_gpt_prompt(d):
    messages_generate = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ]

    if d["extend_instruction"]=="list" and d["id"][-1]!='0':
        template_extend=templatef1.format(original_question=d["conversations"][0]["value"],constraint_list=str(d["constraints"]))

    elif d["extend_instruction"]=="integrate" and d["id"][-1]!='0':
        template_extend=templatef2.format(original_question=d["conversations"][0]["value"],constraint_list=str(d["constraints"]))

    else:
        return
    
    messages_generate[1]["content"] = template_extend

    gpt_prompts.append(messages_generate)

    return gpt_prompts
'''

# generate interaction prompt for constraint extension check


def generate_gpt_prompt(d, gpt_prompts, datadict):

    messages_generate = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ]

    match = re.search(r"【Original Question】:(.*?)\n\s*【Constraint List】",
                      d["messages"][0]["content"], re.DOTALL)
    original_question = match.group(1).strip() if match else None

    original_data = datadict[original_question]
    extend_instruction = original_data["extend_instruction"]

    if len(d["messages"]) <= 1:
        return

    if extend_instruction == 'list':
        match = re.search(r"【Constraint List Data】:(.*)",
                          d["messages"][1]["content"], re.DOTALL)
        extended_question = match.group(1).strip() if match else None
    elif extend_instruction == 'integrate':
        match = re.search(r"【Constraint Integration Format Data】:(.*)",
                          d["messages"][1]["content"], re.DOTALL)
        extended_question = match.group(1).strip() if match else None
    else:
        return

    if not extended_question:
        return

    if extend_instruction == 'list':
        template_check = templatec1.format(
            original_question=original_question, listed_constraint_form_question=extended_question, constraint_list=str(original_data["constraints"]))
    elif extend_instruction == 'integrate':
        template_check = templatec2.format(
            original_question=original_question, integrated_constraint_form_question=extended_question, constraint_list=str(original_data["constraints"]))

    messages_generate[1]["content"] = template_check

    gpt_prompts.append(messages_generate)

    return gpt_prompts


# process interaction results
def extract_data(d, json_data, datadict):
    match = re.search(r"【Original question】:(.*?)\n\s*【",
                      d["messages"][0]["content"], re.DOTALL)
    original_question = match.group(1).strip() if match else None

    original_data = datadict[original_question]
    extend_instruction = original_data["extend_instruction"]

    if len(d["messages"]) <= 1:
        return

    if extend_instruction == 'list':
        match = re.search(r"【Listed constraint form question】:(.*)\n\s*【Constraint list】",
                          d["messages"][0]["content"], re.DOTALL)
        extended_question = match.group(1).strip() if match else None
        # Extract the matched result
        match = re.search(
            r"【Is the listed constraint form question clearly stated to cover all constraints\?】:\s*(Yes|No)", d["messages"][1]["content"])
        if_satisfied = match.group(1).strip() if match else None

    elif extend_instruction == 'integrate':
        match = re.search(r"【Integrated constraint form question】:(.*)\n\s*【Constraint list】",
                          d["messages"][0]["content"], re.DOTALL)
        extended_question = match.group(1).strip() if match else None
        # Extract the matched result
        match = re.search(
            r"【Does the question in the integrated constraint form clearly cover all constraints listed\?】:\s*(Yes|No)", d["messages"][1]["content"])
        if_satisfied = match.group(1).strip() if match else None

    else:
        return

    if not extended_question or not if_satisfied:
        return

    if if_satisfied != 'Yes':
        return

    original_data["conversations"][0]["value"] = extended_question

    json_data.append(original_data)

    return json_data


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', required=True)
    parser.add_argument('--base_url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_interact_file', required=True)
    parser.add_argument('--database_file', required=True)
    parser.add_argument('--res_output_path', required=True)

    args = parser.parse_args()

    return args


def main():
    args = args_parse()

    api_key = args.api_key
    base_url = args.base_url
    model = args.model
    data_interact_file = args.data_interact_file
    database_file = args.database_file
    res_output_path = args.res_output_path

    data_interact = load_jsonl_data(data_interact_file)
    database = load_json_data(database_file)
    datadict = {}
    for d in database:
        datadict[d["conversations"][0]["value"]] = d
    gpt_prompts = []

    for d in data_interact:
        gpt_prompts = generate_gpt_prompt(
            d, gpt_prompts=gpt_prompts, datadict=datadict)

    talker = Talker_GPT(api_key=api_key, base_url=base_url, model=model)
    response = []
    for messages in gpt_prompts:
        response.append(talker.chat(messages))

    json_data = []
    for d in data_interact:
        json_data = extract_data(d, json_data=json_data, datadict=datadict)

    data2json_file(json_data, res_output_path)


if __name__ == '__main__':
    main()
