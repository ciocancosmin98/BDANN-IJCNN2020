SOURCE_TOPIC=politics
TARGET_TOPIC=social
MODEL_PATH="../models/unfrozen_linearlmbd_classic/body_anon_lower_politics-social[hidden-64,lmbd-0.5,bert-unfrozen]/run_04/5-best"
OUTPUT_PATH="../results/politics-social_lmbd-0.5.tsv"

python train_classic.py \
    --source_topic $SOURCE_TOPIC \
    --target_topic $TARGET_TOPIC \
    --hidden_dim 64 \
    --num_epochs 5 \
    --evaluate $MODEL_PATH \
    --save_predictions $OUTPUT_PATH \
    --n_runs 1