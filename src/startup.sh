run_test () {
    TRAIN_TOPICS=$1
    TRAIN_TOPICS_JOIN=`echo $TRAIN_TOPICS | tr " " -`
    TEST_TOPICS=$2
    LAMBDA=$3

    python BDANN_sarcasm.py \
        --train_topics $TRAIN_TOPICS \
        --test_topics $TEST_TOPICS \
        --save_dir ../models/unfrozen_bert_linear_lmbd_0.5/body_anon_lower_$TRAIN_TOPICS_JOIN\[hidden-64\,lmbd-$LAMBDA\,bert-unfrozen] \
        --lmbd $LAMBDA \
        --hidden_dim 64 \
        --num_epochs 5 \
        --n_runs 5
}

# run_test "politics social" "sports" "0.0"
# run_test "politics sports" "social" "0.0"
# run_test "sports social" "politics" "0.0"

run_test "politics social" "sports" "0.5"
run_test "politics sports" "social" "0.5"
run_test "sports social" "politics" "0.5"