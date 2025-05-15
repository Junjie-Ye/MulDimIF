API_KEY="your_openai_api_key"
BASE_URL="your_base_url_if_any"
MODEL="gpt-4o"  # or "gpt-3.5-turbo", etc.

DATA_INTERACT_FILE="path/to/data_interact.jsonl"
DATABASE_FILE="path/to/database.json"
RES_OUTPUT_PATH="path/to/response.json"

# Call the Python script and pass all parameters
python Code/instruction_generation/instruction_rewriting_listing_incorporation.py \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --data_interact_file "$DATA_INTERACT_FILE" \
    --database_file "$DATABASE_FILE" \
    --res_output_path "$RES_OUTPUT_PATH"
