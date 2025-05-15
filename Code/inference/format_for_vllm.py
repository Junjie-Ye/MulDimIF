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


from utils.data_utils import *
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def format_test_data(data_path, save_path=None):
    data = load_data(data_path)
    for item in data:
        for conversation in item['conversations']:
            if 'from' in conversation:
                conversation['role'] = conversation.pop('from')
            if 'value' in conversation:
                conversation['content'] = conversation.pop('value')
    if save_path:
        data2json_file(data=data, file_name=save_path)

    return data
