TRAIN_FILE="Data/train.json"
TEST_FILE="Data/test.json"
LOCAL_DIR="Data"

python Code/rl/data_preprocess/muldimif.py \
    --traindata_path $TRAIN_FILE \
    --testdata_path $TEST_FILE \
    --local_dir $LOCAL_DIR \


