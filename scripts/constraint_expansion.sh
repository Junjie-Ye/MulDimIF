API_KEY="your_openai_api_key"
BASE_URL="your_base_url_if_any"
MODEL="gpt-4o"  # or "gpt-3.5-turbo", etc.

DATA_INTERACT_FILE="path/to/data_interact.jsonl"
DATA_DICT_FILE="path/to/data_dict.json"
NEW_DATA_DICT_FILE="path/to/output_data_dict.json"
RES_OUTPUT_PATH="path/to/response.json"

# Call the Python script and pass all parameters
python Code/instruction_generation/constraint_expansion.py \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --data_interact_file "$DATA_INTERACT_FILE" \
    --data_dict_file "$DATA_DICT_FILE" \
    --new_data_dict_file "$NEW_DATA_DICT_FILE" \
    --res_output_path "$RES_OUTPUT_PATH"