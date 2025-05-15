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
import pandas as pd
from typing import List, Union, Dict
from collections import Counter
import csv
import unicodedata
from openai import OpenAI


def csv_to_xlsx(csv_file_path, xlsx_file_path):
    """
    convert CSV file to XLSX file

    :param csv_file_path: path of CSV file
    :param xlsx_file_path: path of output XLSX file
    """
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8', on_bad_lines='skip')

        df = df.applymap(lambda x: x.replace('\n', ' ').replace(
            '\r', '') if isinstance(x, str) else x)

        df.to_excel(xlsx_file_path, index=False, engine='openpyxl')

        print(f"file has been converted to {xlsx_file_path}")
    except Exception as e:
        print(f"conversion failed: {e}")


def clean_csv(file_path, output_path, encoding='utf-8'):
    """
    clean strange characters in CSV file, and save the cleaned data to a new file.

    Args:
        file_path (str): path of original CSV file.
        output_path (str): path of output CSV file.
        encoding (str): encoding of output file, default is 'utf-8'.
    """
    import chardet

    def detect_encoding(file_path):
        """
        detect the encoding of the file.

        Args:
            file_path (str): path of file.

        Return:
            str: encoding of the file.
        """
        with open(file_path, 'rb') as file:  # open file in binary mode
            raw_data = file.read()  # read the original data of the file
            # use chardet library to detect encoding
            result = chardet.detect(raw_data)
            return result['encoding']  # return the detected encoding

    # call the internal function to detect the encoding of the file
    detected_encoding = detect_encoding(file_path)
    # print the detected encoding
    print(f"the detected encoding of the file is: {detected_encoding}")

    try:
        # open the original file, read the file content with the detected encoding
        with open(file_path, mode='r', encoding=detected_encoding, errors='replace') as file:
            reader = csv.reader(file)  # use csv module to read the file
            cleaned_data = []  # used to store the cleaned data
            for row in reader:  # traverse each row
                cleaned_row = []  # used to store the cleaned row
                for cell in row:  # traverse each cell
                    # replace unprintable characters with '?'
                    cleaned_cell = ''.join(
                        char if char.isprintable() else '?' for char in cell)
                    # add the cleaned cell to the cleaned row
                    cleaned_row.append(cleaned_cell)
                # add the cleaned row to the cleaned data list
                cleaned_data.append(cleaned_row)

        # write the cleaned data to a new file
        with open(output_path, mode='w', encoding=encoding, newline='') as output_file:
            writer = csv.writer(output_file)  # create a csv writer
            # write the cleaned data to the file
            writer.writerows(cleaned_data)

        # print the saved path
        print(f"the cleaned file has been saved to: {output_path}")
    except Exception as e:  # catch the exception
        # print the error information
        print(f"error when processing the file: {e}")


def load_excel_data(file_path, sheet_name=0, usecols=None, na_values=None):
    """
    function to read excel file

    Args:
        file_path (str): path of excel file
        sheet_name (int, optional): name or index of the sheet to read, default is 0 (read the first sheet).
        usecols (list, str, optional): columns to read, default is None (read all columns).
        na_values (list, str, dict, optional): values to be considered as missing, default is None.

    Returns:
        DataFrame or dict: if successfWul, return a DataFrame or a dictionary (multiple sheets).
        None: if failed.
    """
    try:
        # read the excel file
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            usecols=usecols,
            na_values=na_values
        )
        return df
    except FileNotFoundError:
        print(f"error: file not found, please check the path -> {file_path}")
        return None
    except Exception as e:
        print(f"error when reading the file: {e}")
        return None


def load_dir_path(dir):
    path = []
    for i in os.listdir(dir):
        pth = os.path.join(dir, i)
        path.append(pth)
    return path


def load_csv_data(csv_file_path, exist_head=True):
    """load csv file

    Args:
        csv_file_path (str): path of csv file
        exist_head (bool, optional): whether the file has a header, default is True.

    Returns:
        dict: return a dictionary data_dict = {"head":None, "data":[]}
    """
    data_dict = {"head": None, "data": []}
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:
        # create a csv reader
        reader = csv.reader(file)

        # traverse each row in the csv file
        for index, row in enumerate(reader):
            if exist_head and index == 0:
                data_dict['head'] = row
                continue
            data_dict['data'].append(row)

    return data_dict


def get_csv_length(file_path):
    """
    get the total number of rows in the csv file (including the header row).

    :param file_path: path of csv file
    :return: total number of rows in the file
    """
    with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:
        return sum(1 for _ in file)


def data2csv(data, file_name, head=None, mode="w"):
    """output data to csv file

    Args:
        data (list[list]): data list
        file_name (str): path of output file
        head (list, optional): header of the file, default is None.
        mode (str, optional): writing mode, default is "w".
    """

    with open(file_name, mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if head is not None:
            writer.writerow(head)

        for row in data:
            writer.writerow(row)


def get_files(folder_path, file_extension):
    """get files with specified extension in the folder

    Args:
        folder_path (str): path of folder
        file_extension (str): file extension

    Returns:
        List[str]: return a list of files with the specified extension
    """
    files = []

    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith("." + file_extension):
                files.append(os.path.join(root, filename))

    return files


def load_json_data(json_file_path) -> list:
    try:
        # open and read the json file
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            return data

    except FileNotFoundError:
        print(f"file not found: {json_file_path}")
    except json.JSONDecodeError:
        print(f"error when parsing the json file: {json_file_path}")


def load_jsonl_data(jsonl_file_path) -> list:
    data = []
    try:
        with open(jsonl_file_path, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"file not found: {jsonl_file_path}")
    except json.JSONDecodeError:
        print(f"error when parsing the json file: {jsonl_file_path}")


def data2json_file(data, file_name, mode="w"):
    """output data to json file

    Args:
        data (Any): data to be output
        file_name (str): path of json file
        mode(str): output mode, default is "w" write mode, can be changed to "a" append mode, etc.
    """
    # check if the path exists, if not, create it
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # use with statement to open the file, and write the data to the file
    with open(file_name, mode=mode) as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)


def parquet2json(parquet_file: str, output_path: str):
    """convert parquet file to json file

    Args:
        parquet_file (str): path of parquet file
        output_path (str): path of output file (not including the file name)
    """
    # read the parquet file
    df = pd.read_parquet(parquet_file)

    # get the file name
    output_file_name = parquet_file[parquet_file.rfind("/") + 1:]
    output_file_name = output_file_name[: output_file_name.find("-")]
    output_file = f"{output_path}/{output_file_name}.json"

    # output the result to the json file
    df.to_json(output_file, orient="records")


def jsonl2json(jsonl_file: str, json_file: str):
    # open the input file and output file
    with open(jsonl_file, "r", encoding="utf-8") as input_file:
        # read the jsonl file line by line, and parse each line to a python dictionary
        data = [json.loads(line) for line in input_file]
        # write each json object to the output file
    data2json_file(data, json_file)
    print(f"successfully converted {jsonl_file} to {json_file}")


def data_non_exist(data_name=""):
    print(f"{data_name} data does not exist!")


def data2jsonl_file(data: Union[List[Dict], Dict], file_name, mode="w"):
    """output data to json file

    Args:
        data (Any): data to be output (can be passed in a list or a single dictionary data)
        file_name (str): path of jsonl file
        mode(str): output mode, default is "w" write mode, can be changed to "a" append mode, etc.
    """
    with open(file_name, mode=mode, encoding="utf-8") as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"file {file_path} has been deleted.")
    else:
        print(f"file {file_path} does not exist.")


def load_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    # first check by extension
    if file_extension == ".json":
        return load_json_data(file_path)
    elif file_extension == ".jsonl":
        return load_jsonl_data(file_path)
    elif file_extension == ".csv":
        return load_csv_data(file_path)
    elif file_extension == ".xlsx":
        return load_excel_data(file_path)
    else:
        return "Unknown file type"


class Talker_GPT:
    def __init__(self, api_key=None, base_url=None, model=None) -> None:
        self.api_key = None  # "sk-xxxx"
        self.base_url = None  # "https://api.smart-gpt-xxx" # leave blank if not available
        self.model = None

        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = base_url
        if model:
            self.model = model

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def set_model(self, model):
        self.model = model

    def chat(self, messages):
        assert self.model, "please set model"

        if not messages:
            # This is the message for a single-turn conversation
            # For multi-turn conversation, keep adding "assistant" replies and "user" messages alternately
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]

        chat_completion = self.client.chat.completions.create(
            model=self.model,  # The model to use
            messages=messages,
            temperature=0.7,
            max_completion_tokens=2048
        )

        res = chat_completion.choices[0].message
        res = unicodedata.normalize('NFKC', res.content)
        return res
