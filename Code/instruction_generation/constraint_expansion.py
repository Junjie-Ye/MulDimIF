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


import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # NOQA
from Code.utils.data_utils import *
import time
import re
import argparse


constraints = [
    '''
	Main Category : Language
	Subcategory : {
	Chinese: Simplified, Traditional
    English: All Uppercase, Capitalized, Lowercase
	}''',
    '''
	Main Category : Format
	Subcategory : {
	Markdown: Heading levels, Block quotes
    Json: Object nesting levels
    XML: Number of attributes
    Table: Row limit, Column limit
	}''',
    '''
	Main Category : Length
	Subcategory : {
	Words: At most, At least, Range
    Sentences: At most, At least, Range
    Paragraphs: At most, At least, Range
	}''',
    '''
	Main Category : Content
	Subcategory : {
    Keywords: Must include, Repeated, Avoid
    Identifiers: Start identifier, End identifier, Delimiting identifier
    Punctuation: Ending punctuation, Exclude punctuation
	}'''
]

template_g = '''You are an expert in instruction-following data construction. Your task is to generate corresponding data as required.

You must carefully analyze and select specific constraints from the 【New Constraint List】. Then, based on the original question in the provided 【Data】, generate new data that adheres to the 【Data Generation Requirements】. Finally, respond in the format specified in the 【Response Format】.

【New Constraint List】: {new_constraint_list}

【Data Generation Requirements】:

    【Core Requirements】:

        1. Ensure only {c1} is added, that is, {c2}. The word following 【Main Category】 should be the main category.

        2. Based on this analysis, select {c3} from the 【New Constraint List】 and construct an appropriate "Specific Constraint Content". Add it to the 【Original Constraint List】 in the provided data, and return the 【Updated Constraint List】.

        3. Modify the content of the 【Original Question】 in the provided data to **explicitly and clearly specify all the constraints** in the 【Updated Constraint List】. The modified question must clearly describe each constraint in natural language, ensuring that the constraints are fully integrated into the question text. For example:
           - Original Question: "Tell me about machine learning."
           - Constraint: "The answer must use capitalized letters for each word."
           - Modified Question: "Tell me about machine learning. The answer must use capitalized letters for each word."

        4. Ensure that the Specific Constraint in each constraint triplet is detailed and specific, containing concrete information or examples (e.g., instead of "Must include", specify "Must include the keyword 'machine learning'").
    
    【Notes】:

        1. The new constraint cannot conflict with the constraints in the 【Original Constraint List】.

        2. The modified 【Question with the New Constraint】 must **explicitly describe all the constraints** in natural language, ensuring that the constraints are fully integrated into the question text. Constraints should not be implicitly applied to the answer without being explicitly stated in the question.

        3. Make sure the Specific Constraint in each constraint triplet is as specific as possible, including concrete details or examples.

        4. **Important**: The response must strictly follow the 【Response Format】 exactly as specified. Do not include any numbering, bullet points, or additional formatting. The 【Updated Constraint List】 must be outputted as a single list of tuples in the exact format shown, without any additional characters or line breaks between the tuples.

        5. When generating the modified 【Question with the New Constraint】, ensure that the language is natural and well-polished. Enrich the phrasing of constraints to avoid redundancy and monotony.

【Response Format】:

    【Thinking Process】: xxx

    【Updated Constraint List】: [(Main Category, Subcategory, Specific Constraint), (Main Category, Subcategory, Specific Constraint), ...]  (The main category is the word after 【Main Category】, and the constraints we provide are just broad scopes. You need to find suitable specific constraints based on the question and its answers. The Specific Constraint should be detailed and specific.)

    【Question with the New Constraint】: xxx

【Data】:

    【Original Constraint List】: [{original_constraint_list}]

    【Original Question】: {original_question}'''


template_c = '''You are an expert in data structure following instructions. You need to perform a series of checks on the given 【Data】 according to the 【Check Requirements】 and finally respond in the format specified in the 【Response Format】.

【Check Requirements】:
    1. Check if there is any constraint conflict in the "Constraint List" in the provided data. Explain first and then conclude.
    2. Check if the "Question" in the provided data clearly specifies all the constraint requirements in the "Constraint List". Explain first and then conclude.
    3. The response format should follow the requirements specified in the 【Response Format】 below.

【Response Format】:
    # Constraint Conflict Check #
    【Specific Explanation】:
    【Is there any constraint conflict in the constraints of the data】: [Yes/No]

    # Does the Question clearly specify all constraints in the Constraint List Check #
    【Specific Explanation】: [Explanation]
    【Does the question include all constraints from the constraint list】: [Yes/No]

【Data】:
    【Constraint List】: [{constraint_list}]
    【Question】: {quetsion}'''


def generate_template_generate(original_constraint_list, original_question, constraint_set):
    random.seed(time.time())
    new_constraint_list = random.choice(list(constraint_set))
    constraint_set.remove(new_constraint_list)
    random.seed(time.time())
    number = random.choice([1, 2])
    if 'Language' in new_constraint_list:
        number = 1
    if number == 1:
        c1 = "one new constraint is"
        c2 = "a single (Primary Category, Secondary Category, Specific Constraint) triplet"
        c3 = "one constraint"
    else:
        c1 = "two new constraints are"
        c2 = "two (Primary Category, Secondary Category, Specific Constraint) triplets"
        c3 = "two constraints"

    template_generate = template_g.format(new_constraint_list=new_constraint_list, c1=c1, c2=c2,
                                          c3=c3, original_constraint_list=original_constraint_list, original_question=original_question)
    return template_generate, new_constraint_list


def generate_template_check(constraint_list, quetsion):

    template_check = template_c.format(
        constraint_list=constraint_list, quetsion=quetsion)

    return template_check


def extract_generate(response):

    updated_constraint_list = re.search(
        r'【Updated Constraint List】\:\s*\[(.*?)\]', response, re.DOTALL)
    updated_constraint_list = updated_constraint_list.group(
        1).strip() if updated_constraint_list else None

    question_with_new_constraint = re.search(
        r'【Question with the New Constraint】\:\s*(.*)', response, re.DOTALL)
    question_with_new_constraint = question_with_new_constraint.group(
        1).strip() if question_with_new_constraint else None

    return updated_constraint_list, question_with_new_constraint


def extract_check(response):

    if_constraint_conflict = re.search(
        r'【Is there any constraint conflict in the constraints of the data】\:\s*(No|Yes)', response, re.DOTALL)
    if_constraint_conflict = if_constraint_conflict.group(
        1).strip() if if_constraint_conflict else None

    if_question_include_constraint = re.search(
        r'【Does the question include all constraints from the constraint list】\:\s*(No|Yes)', response, re.DOTALL)
    if_question_include_constraint = if_question_include_constraint.group(
        1).strip() if if_question_include_constraint else None

    return if_constraint_conflict, if_question_include_constraint


def inclusion(str1, str2):
    # split processing Chinese and English parentheses and commas
    str1_list = re.split(r'\)，\s*\（|\),\s*\(', str1)
    str2_list = re.split(r'\)，\s*\（|\),\s*\(', str2)

    # remove extra parentheses and whitespace characte  rs
    for i in range(len(str1_list)):
        str1_list[i] = re.sub(r'[()（）]', '', str1_list[i].strip())

    for i in range(len(str2_list)):
        str2_list[i] = re.sub(r'[()（）]', '', str2_list[i].strip())

    # convert to set for inclusion judgment
    str1_set = set(str1_list)
    if str1 == "":
        str1_set = set()
    str2_set = set(str2_list)

    return str1_set.issubset(str2_set)


def parse_constraints(constraint_string):
    # split processing Chinese and English parentheses and commas
    items = re.split(r'\)，\s*\（|\),\s*\(', constraint_string)

    result = []
    for item in items:
        item = item.strip("()（）")  # remove parentheses
        # match the content after the first comma, including Chinese and English commas
        parts = re.split(r'[，,]', item, 2)
        result.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))

    return result


def generate_gpt_prompt(d, gpt_prompts, data_dict):
    # process interaction results of constraint generate
    messages_generate = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ]

    match = re.search(r'【Original Question】: (.*)',
                      d["messages"][0]["content"], re.DOTALL)
    original_question = match.group(1).strip() if match else None

    j = None
    for i in range(len(data_dict)):
        if original_question == data_dict[i]["original_question_s"][-1].strip():
            j = i
            break
    if j == None:
        return

    if len(d["messages"]) <= 1:
        return

    updated_constraint_list, question_with_new_constraint = extract_generate(
        d['messages'][1]['content'])

    if not updated_constraint_list or not question_with_new_constraint:
        return

    data_dict[j]["original_question_s"].append(question_with_new_constraint)
    data_dict[j]["original_constraint_list_s"].append(updated_constraint_list)

    template_check = generate_template_check(
        updated_constraint_list, question_with_new_constraint)

    messages_generate[1]["content"] = template_check

    gpt_prompts.append(messages_generate)

    return gpt_prompts


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', required=True)
    parser.add_argument('--base_url', default=None)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_interact_file', required=True)
    parser.add_argument('--data_dict_file', required=True)
    parser.add_argument('--new_data_dict_file', required=True)
    parser.add_argument('--res_output_path', required=True)

    args = parser.parse_args()

    return args


def main():
    args = args_parse()

    api_key = args.api_key
    base_url = args.base_url
    model = args.model
    data_interact_file = args.data_interact_file
    data_dict_file = args.data_dict_file
    new_data_dict_file = args.new_data_dict_file
    res_output_path = args.res_output_path

    data_interact = load_jsonl_data(data_interact_file)
    gpt_prompts = []
    data_dict = load_json_data(data_dict_file)

    for d in data_interact:
        gpt_prompts = generate_gpt_prompt(
            d, gpt_prompts=gpt_prompts, data_dict=data_dict)

    data2json_file(data_dict, new_data_dict_file)

    talker = Talker_GPT(api_key=api_key, base_url=base_url, model=model)
    response = []
    for messages in gpt_prompts:
        response.append(talker.chat(messages))
    data2json_file(response, res_output_path)


if __name__ == '__main__':
    main()
