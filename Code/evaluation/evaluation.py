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
import sys
print("Python version:", sys.version)  # NOQA
from collections import defaultdict
from pprint import pprint
from scripts import *
from utils.data_utils import *
import argparse


class_mapping = {
    'Content_Keywords': Content_Keywords(),
    'Content_Keywords: Must include': Content_Keywords(),
    'Length_Words': Length_Words(),
    'Length_Words: At most': Length_Words(),
    'Length_Words: At least': Length_Words(),
    'Length_Words: Range': Length_Words(),
    'Length_Sentences': Length_Sentences(),
    'Length_Sentences: At least': Length_Sentences(),
    'Length_Sentences: At most': Length_Sentences(),
    'Length_Sentences: Range': Length_Sentences(),
    'Length_Paragraphs': Length_Paragraphs(),
    'Length_Paragraphs: At most': Length_Paragraphs(),
    'Length_Paragraphs: At least': Length_Paragraphs(),
    'Length_Paragraphs: Range': Length_Paragraphs(),
    'Paragraphs_At most': Length_Paragraphs(),
    'Format_Table': Format_Table(),
    'Table_Row limit': Format_Table(),
    'Table_Column limit': Format_Table(),
    'Format_Table: Row limit': Format_Table(),
    'Format_Table: Column limit': Format_Table(),
    'Punctuation_Ending punctuation': Content_Punctuation(),
    'Content_Punctuation: Ending punctuation': Content_Punctuation(),
    'Content_Punctuation': Content_Punctuation(),
    'Language_English': Language_English(),
    'Language_English: Capitalized': Language_English(),
    'Language_English: All Uppercase': Language_English(),
    'Format_Markdown': Format_Markdown(),
    'Markdown_Heading levels': Format_Markdown(),
    'Format_Markdown: Heading levels': Format_Markdown(),
    'Markdown_Block quotes': Format_Markdown(),
    'Json_Object nesting levels': Format_Json(),
    'Format_Json': Format_Json(),
    'Language_Chinese': Language_Chinese(),
    'Language_Chinese: Simplified': Language_Chinese(),
    'Language_Chinese: Traditional': Language_Chinese(),
    'Content_Identifiers': Content_Others(),
    'Content_Length': Content_Others(),
    'Citations_In-text': Content_Others(),
    'Content_Quotes': Content_Others(),
    'Content_Sources': Content_Others(),
    'Content_Mention': Content_Others(),
    'Format_Markdown: Block quotes': Format_Others(),
    'Format_Text': Format_Others(),
    'XML_Number of attributes': Format_Others(),
    'References_Format': Format_Others(),
    'Format_Bullet Points': Format_Others(),
    'Format_XML': Format_Others(),
    'Format_Blurb': Format_Others(),
    'Table_Table': Format_Others(),
    'Sentences_At most': Length_Sentences(),
    'Sentences_At least': Length_Sentences(),
    'Words_At most': Length_Words(),
    'Json_Number of attributes': Format_Json(),
    'Format_Word Count': Length_Words(),
    'Format_Length': Format_Others(),
}


# make the mapping from the first and second level constraints to the original 4 first level and 12 second level constraints
constraint_mapping = {
    'Content_Keywords': 'Content_Keywords',
    'Content_Keywords: Must include': 'Content_Keywords',
    'Length_Words': 'Length_Words',
    'Length_Words: At most': 'Length_Words',
    'Length_Words: At least': 'Length_Words',
    'Length_Words: Range': 'Length_Words',
    'Words_At most': 'Length_Words',
    'Length_Sentences': 'Length_Sentences',
    'Length_Sentences: At least': 'Length_Sentences',
    'Length_Sentences: At most': 'Length_Sentences',
    'Length_Sentences: Range': 'Length_Sentences',
    'Sentences_At most': 'Length_Sentences',
    'Sentences_At least': 'Length_Sentences',
    'Length_Paragraphs': 'Length_Paragraphs',
    'Length_Paragraphs: At most': 'Length_Paragraphs',
    'Length_Paragraphs: At least': 'Length_Paragraphs',
    'Length_Paragraphs: Range': 'Length_Paragraphs',
    'Paragraphs_At most': 'Length_Paragraphs',
    'Format_Table': 'Format_Table',
    'Table_Row limit': 'Format_Table',
    'Table_Column limit': 'Format_Table',
    'Format_Table: Row limit': 'Format_Table',
    'Format_Table: Column limit': 'Format_Table',
    'Punctuation_Ending punctuation': 'Content_Punctuation',
    'Content_Punctuation: Ending punctuation': 'Content_Punctuation',
    'Content_Punctuation': 'Content_Punctuation',
    'Language_English': 'Language_English',
    'Language_English: Capitalized': 'Language_English',
    'Language_English: All Uppercase': 'Language_English',
    'Format_Markdown': 'Format_Markdown',
    'Markdown_Heading levels': 'Format_Markdown',
    'Format_Markdown: Heading levels': 'Format_Markdown',
    'Markdown_Block quotes': 'Format_Markdown',
    'Json_Object nesting levels': 'Format_Json',
    'Format_Json': 'Format_Json',
    'Language_Chinese': 'Language_Chinese',
    'Language_Chinese: Simplified': 'Language_Chinese',
    'Language_Chinese: Traditional': 'Language_Chinese',
    'Content_Identifiers': 'Content_Identifiers',
    'Content_Length': 'Length_Words',
    'Citations_In-text': 'Content_Identifiers',
    'Content_Quotes': 'Content_Punctuation',
    'Content_Sources': 'Content_Identifiers',
    'Content_Mention': 'Content_Keywords',
    'Format_Markdown: Block quotes': 'Format_Markdown',
    'Format_Text': 'Content_Identifiers',
    'XML_Number of attributes': 'Format_XML',
    'References_Format': 'Content_Identifiers',
    'Format_Bullet Points': 'Content_Identifiers',
    'Format_XML': 'Format_XML',
    'Format_Blurb': 'Length_Words',
    'Table_Table': 'Format_Table',
    'Json_Number of attributes': 'Format_Json',
    'Format_Word Count': 'Length_Words',
    'Format_Length': 'Length_Words',
}


# use variable selection class
def get_instance(class_name):
    cls = class_mapping.get(class_name)
    if cls:
        return cls
    else:
        raise ValueError(f"Class '{class_name}' not found")

# data pre-processing


def pre_process(data, mode_type):
    if mode_type == 'deepseek':
        new_data = []
        for d in data:
            res = d['conversations'][-1]['content']
            # find the position of the last </think>, remove the content before it, and the following newline characters
            last_think_index = res.rfind('</think>')
            if last_think_index != -1:
                res = res[last_think_index + len('</think>'):]
                # remove all leading newline characters
                res = res.strip('\n')
            d['conversations'][-1]['content'] = res
            new_data.append(d)
        return new_data
    else:
        return data

# map the first and second level constraints to the original 4 first level and 12 second level constraints


def map_constraint(data):
    new_data = []
    for d in data:
        new_constraints = []
        for constraint in d['constraints']:
            key = f"{constraint[0]}_{constraint[1]}"
            value = constraint_mapping[key]
            first, second = value.split('_')
            new_constraint = [
                first,
                second,
                constraint[-1]
            ]
            new_constraints.append(new_constraint)
        d['constraints'] = new_constraints
        new_data.append(d)
    return new_data

# calculate the score


def get_score(data):
    # map the first and second level constraints to the original 4 first level and 12 second level constraints
    data = map_constraint(data)

    # ====================== calculate the overall score ======================
    num_data = len(data)  # the total length of data
    num_constraint = 0  # the total number of constraints
    total_acc = 0  # 01 scoring
    total_acc_macro = 0  # macro average fine-grained scoring
    total_acc_micro = 0  # micro average fine-grained scoring
    for item in data:
        judges = item['judges']
        num_constraint += len(judges)
        if sum(judges) == len(judges):
            total_acc += 1  # if all correct, acc+1
        # macro average single item
        total_acc_macro += sum(judges) / len(judges)
        total_acc_micro += sum(judges)  # micro average single item
    total_acc = f"{total_acc}/{num_data}={total_acc/num_data}"
    total_acc_macro = f"{total_acc_macro}/{num_data}={total_acc_macro/num_data}"
    total_acc_micro = f"{total_acc_micro}/{num_constraint}={total_acc_micro/num_constraint}"

    # ====================== calculate the score of each constraint extension form ======================
    constraint_extension_list = defaultdict(
        int)  # 'list', 'integrate', 'example'
    constraint_extension_list_num = defaultdict(int)
    constraint_extension_list_macro = defaultdict(int)
    constraint_extension_list_micro = defaultdict(int)
    constraint_extension_list_micro_num = defaultdict(int)

    for item in data:
        judges = item['judges']
        constraint_extension = item['constraint_pattern']

        constraint_extension_list_num[constraint_extension] += 1
        if sum(judges) == len(judges):
            constraint_extension_list[constraint_extension] += 1

        constraint_extension_list_macro[constraint_extension] += sum(
            judges) / len(judges)

        constraint_extension_list_micro_num[constraint_extension] += len(
            judges)
        constraint_extension_list_micro[constraint_extension] += sum(judges)

    # calculate the score of each constraint extension form
    constraint_extension_list = {
        k: f"{v}/{constraint_extension_list_num[k]}={v/constraint_extension_list_num[k]}" for k, v in constraint_extension_list.items()}
    constraint_extension_list_macro = {
        k: f"{v}/{constraint_extension_list_num[k]}={v/constraint_extension_list_num[k]}" for k, v in constraint_extension_list_macro.items()}
    constraint_extension_list_micro = {
        k: f"{v}/{constraint_extension_list_micro_num[k]}={v/constraint_extension_list_micro_num[k]}" for k, v in constraint_extension_list_micro.items()}

    # ====================== calculate the score of each constraint type ======================
    # constraint_type_list = defaultdict(int)
    # constraint_type_num_list = defaultdict(int)
    # for item in data:
    #     for constraint, judge in zip(item['constraints'], item['judges']):
    #         constraint_type = constraint[0]
    #         constraint_type_num_list[constraint_type] += 1
    #         constraint_type_list[constraint_type] += judge
    # constraint_type_list = {k: f"{v}/{constraint_type_num_list[k]}={v/constraint_type_num_list[k]}" for k, v in constraint_type_list.items()}

    constraint_type_list = defaultdict(int)
    constraint_type_num_list = defaultdict(int)
    for item in data:
        cnt = defaultdict(list)
        for constraint, judge in zip(item['constraints'], item['judges']):
            cnt[constraint[0]].append(judge)
        for constraint_type, judges in cnt.items():
            constraint_type_num_list[constraint_type] += 1
            if sum(judges) == len(judges):
                constraint_type_list[constraint_type] += 1
    constraint_type_list = {
        k: f"{v}/{constraint_type_num_list[k]}={v/constraint_type_num_list[k]}" for k, v in constraint_type_list.items()}

    # ====================== calculate the score of each second level constraint type ======================
    constraint_type_second_list = defaultdict(int)
    constraint_type_second_num_list = defaultdict(int)
    for item in data:
        for constraint, judge in zip(item['constraints'], item['judges']):
            constraint_type_second = f"{constraint[0]}_{constraint[1]}"
            constraint_type_second_num_list[constraint_type_second] += 1
            constraint_type_second_list[constraint_type_second] += judge
    constraint_type_second_list = {
        k: f"{v}/{constraint_type_second_num_list[k]}={v/constraint_type_second_num_list[k]}" for k, v in constraint_type_second_list.items()}

    # ====================== calculate the score of each constraint difficulty ======================
    constraint_difficulty_list = defaultdict(int)
    constraint_difficulty_list_num = defaultdict(int)
    constraint_difficulty_list_macro = defaultdict(int)
    constraint_difficulty_list_micro = defaultdict(int)
    constraint_difficulty_list_micro_num = defaultdict(int)

    for item in data:
        judges = item['judges']
        constraint_difficulty = item['difficulty']

        constraint_difficulty_list_num[constraint_difficulty] += 1
        if sum(judges) == len(judges):
            constraint_difficulty_list[constraint_difficulty] += 1

        constraint_difficulty_list_macro[constraint_difficulty] += sum(
            judges) / len(judges)

        constraint_difficulty_list_micro_num[constraint_difficulty] += len(
            judges)
        constraint_difficulty_list_micro[constraint_difficulty] += sum(judges)

    # calculate the score of each constraint difficulty
    constraint_difficulty_list = {
        k: f"{v}/{constraint_difficulty_list_num[k]}={v/constraint_difficulty_list_num[k]}" for k, v in constraint_difficulty_list.items()}
    constraint_difficulty_list_macro = {
        k: f"{v}/{constraint_difficulty_list_num[k]}={v/constraint_difficulty_list_num[k]}" for k, v in constraint_difficulty_list_macro.items()}
    constraint_difficulty_list_micro = {
        k: f"{v}/{constraint_difficulty_list_micro_num[k]}={v/constraint_difficulty_list_micro_num[k]}" for k, v in constraint_difficulty_list_micro.items()}
    # sort the difficulty data by key
    constraint_difficulty_list = dict(
        sorted(constraint_difficulty_list.items()))
    constraint_difficulty_list_macro = dict(
        sorted(constraint_difficulty_list_macro.items()))
    constraint_difficulty_list_micro = dict(
        sorted(constraint_difficulty_list_micro.items()))

    # ====================== summarize the above scores ======================

    score = {
        'constraint_pattern_list': constraint_extension_list,
        'constraint_category_list': constraint_type_list,
        'constraint_category_second_list': constraint_type_second_list,
        'constraint_difficulty_list': constraint_difficulty_list,
        'Overall': total_acc,

    }

    return score


# check the situation of each case of data
def check(data):
    judge_data = []
    for item in data:
        res = item['conversations'][-1]['content']
        item['judges'] = []
        for constraint in item['constraints']:
            cls_name = f"{constraint[0]}_{constraint[1]}"
            judge_result = get_instance(cls_name).check(constraint[-1], res)
            judge_result = 1 if judge_result else 0
            item['judges'].append(judge_result)
        judge_data.append(item)
    return judge_data

# main entrance


def eval_by_code(data_path, mode_type, save_path=None):
    data = load_data(data_path)
    data = pre_process(data, mode_type)
    judge_data = check(data)
    score = get_score(judge_data)
    if save_path:
        # output to file
        data2json_file(score, save_path)
    pprint(score)


# example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str,
                        default="Data/grpo_deepseekr1_llama-8b.jsonl")
    parser.add_argument("--save_path", type=str,
                        default="Data/grpo_deepseekr1_llama-8b.json")
    args = parser.parse_args()
    # read all jsonl files in the folder, corresponding to generating score and saving
    file_path = args.file_path
    mode_type = 'deepseek' if 'DeepSeek' in file_path or 'deepseek' in file_path else 'auto'
    save_path = args.save_path
    eval_by_code(data_path=file_path, mode_type=mode_type, save_path=save_path)
