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
import re
import datasets
import argparse
from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindata_path', default="Data/train.json")
    parser.add_argument('--testdata_path', default="Data/test.json")
    parser.add_argument('--local_dir', default="Data")
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    dataset = datasets.load_dataset(
        'json',
        data_files={
            'train': args.traindata_path,
            'test': args.testdata_path
        }
    )

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id

    def make_map_fn(split):
        def process_fn(example, idx):
            idx = example.pop('id')
            conversations = example.pop('conversations')
            question = conversations[0]['content']
            # answer = conversations[1]['content']

            constraints = example.pop('constraints')
            data = {
                "data_source": "muldimif",
                "idx": idx,
                "split": split,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                # "response": answer,
                "constraints": constraints
            }
            if idx == "07dfd2e372030f83741e9347a2dde3cc-1":
                print(data)
            return data

        return process_fn
    train_dataset = train_dataset.map(
        function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(
        function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
