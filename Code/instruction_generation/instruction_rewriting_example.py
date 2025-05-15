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
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # NOQA
from Code.utils.data_utils import *
import argparse


def add_examples(value, examples, number_examples):
    s = ""
    for i in range(len(examples)):
        example = examples[i]
        s = ("# Example "+str(3-number_examples-i)+":\n")+("**Question**: " +
                                                           example["conversations"][0]["value"]+"\n")+("**Answer**: "+example["conversations"][1]["value"]+"\n")+"\n"+s

    return s+value


def generate_example_data(data, output_data, database):
    for d in data:
        id = d["id"]
        used_id = [id]
        num_examples = 0

        d["conversations"][0]["value"] = "**Question**: " + \
            d["conversations"][0]["value"]

        constraints1 = [c[1].strip("'\"") for c in d["constraints"]]
        shot1 = []
        for db in database:
            if [c[1].strip("'\"") for c in db["constraints"]] == constraints1 and db["id"] not in used_id:
                shot1.append(db)

        number1 = min(3-num_examples, len(shot1))

        examples = random.sample(shot1, number1)
        for example in examples:
            used_id.append(example["id"])

        d["conversations"][0]["value"] = add_examples(
            d["conversations"][0]["value"], examples, num_examples)
        num_examples += number1

        if num_examples >= 3:
            output_data.append(d)
            continue

        helper = [c[0].strip("'\"") for c in d["constraints"]]
        k = 0
        while (1):
            if helper[k] == helper[-1]:
                break
            k += 1

        constraints2 = [c[1].strip("'\"") for c in d["constraints"]]
        constraints2_1 = sorted(constraints2[0:k])
        constraints2_2 = constraints2[k:]

        shot2 = []
        for db in database:
            helper = [c[0].strip("'\"") for c in db["constraints"]]
            k = 0
            while (1):
                if helper[k] == helper[-1]:
                    break
                k += 1

            constraints22 = [c[1].strip("'\"") for c in db["constraints"]]
            constraints22_1 = sorted(constraints22[0:k])
            constraints22_2 = constraints22[k:]

            if constraints22_1 == constraints2_1 and constraints22_2 == constraints2_2 and db["id"] not in used_id:
                shot2.append(db)

        number2 = min(3-num_examples, len(shot2))

        examples = random.sample(shot2, number2)
        for example in examples:
            used_id.append(example["id"])

        d["conversations"][0]["value"] = add_examples(
            d["conversations"][0]["value"], examples, num_examples)
        num_examples += number2

        if num_examples >= 3:
            output_data.append(d)
            continue

        helper = [c[0].strip("'\"") for c in d["constraints"]]
        k = 0
        while (1):
            if helper[k] == helper[-1]:
                break
            k += 1

        constraints3 = [c[1].strip("'\"") for c in d["constraints"]]
        constraints3_1 = helper[0:k]
        constraints3_2 = constraints3[k:]

        shot3 = []
        for db in database:
            helper = [c[0].strip("'\"") for c in db["constraints"]]
            k = 0
            while (1):
                if helper[k] == helper[-1]:
                    break
                k += 1

            constraints33 = [c[1].strip("'\"") for c in db["constraints"]]
            constraints33_1 = helper[0:k]
            constraints33_2 = constraints33[k:]

            if constraints3_1 == constraints33_1 and constraints3_2 == constraints33_2 and db["id"] not in used_id:
                shot3.append(db)

        number3 = min(3-num_examples, len(shot3))

        examples = random.sample(shot3, number3)
        for example in examples:
            used_id.append(example["id"])

        d["conversations"][0]["value"] = add_examples(
            d["conversations"][0]["value"], examples, num_examples)
        num_examples += number3

        if num_examples >= 3:
            output_data.append(d)
            continue

        helper = [c[0].strip("'\"") for c in d["constraints"]]
        k = 0
        while (1):
            if helper[k] == helper[-1]:
                break
            k += 1

        constraints4 = [c[1].strip("'\"") for c in d["constraints"]]
        constraints4_1 = []
        for con in helper[0:k]:
            if con not in constraints4_1:
                constraints4_1.append(con)

        constraints4_2 = constraints4[k:]

        shot4 = []
        for db in database:
            helper = [c[0].strip("'\"") for c in db["constraints"]]
            k = 0
            while (1):
                if helper[k] == helper[-1]:
                    break
                k += 1

            constraints44 = [c[1].strip("'\"") for c in db["constraints"]]
            constraints44_1 = []
            for con in helper[0:k]:
                if con not in constraints44_1:
                    constraints44_1.append(con)

            constraints44_2 = constraints44[k:]

            if constraints4_1 == constraints44_1 and constraints4_2 == constraints44_2 and db["id"] not in used_id:
                shot4.append(db)

        number4 = min(3-num_examples, len(shot4))

        examples = random.sample(shot4, number4)
        for example in examples:
            used_id.append(example["id"])

        d["conversations"][0]["value"] = add_examples(
            d["conversations"][0]["value"], examples, num_examples)
        num_examples += number4

        if num_examples >= 3:
            output_data.append(d)
            continue

    return output_data


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data_path", required=True)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()

    return args


def main():
    args = args_parse()

    source_data_path = args.source_data_path
    output_path = args.output_path

    total_data = load_json_data(source_data_path)
    database = [copy.deepcopy(td) for td in total_data if td["extend_instruction"]
                == 'example' and td["difficulty"][-1] != '0' and len(td["conversations"]) >= 2]
    data = [copy.deepcopy(td) for td in total_data if td["extend_instruction"] ==
            'example' and td["difficulty"][-1] != '0' and len(td["conversations"]) >= 2]
    output_data = []

    output_data = generate_example_data(data, output_data, database)

    data2json_file(output_data, output_path)


if __name__ == "__main__":
    main()
