run_test () {
    TRAIN_TOPICS=$1
    TEST_TOPICS=$2
    LAMBDA=$3
    SAVE_DIR=$4
    EXTRA_OPTIONS=$5

    TRAIN_JOINT=`echo $TRAIN_TOPICS | sed 's/ /-/g'`

    python train_bdann.py \
        --train_topics $TRAIN_TOPICS \
        --test_topics $TEST_TOPICS \
        --save_dir $SAVE_DIR/body_anon_lower_$TRAIN_JOINT\[hidden-64\,lmbd-$LAMBDA\,bert-unfrozen] \
        --lmbd $LAMBDA \
        --hidden_dim 64 \
        --num_epochs 5 \
        $EXTRA_OPTIONS \
        --n_runs 5
}

LMBD1="0.5"
LMBD2="1.0"
SDIR="../models/unfrozen_linearlmbd_fullyunsup"

mkdir $SDIR

run_test "politics social" "sports" "0.0"  $SDIR
run_test "politics social" "sports" $LMBD1  $SDIR
run_test "politics social" "sports" $LMBD2  $SDIR

run_test "politics sports" "social" "0.0"  $SDIR
run_test "politics sports" "social" $LMBD1  $SDIR
run_test "politics sports" "social" $LMBD2  $SDIR

run_test "social sports" "politics" "0.0"  $SDIR
run_test "social sports" "politics" $LMBD1  $SDIR
run_test "social sports" "politics" $LMBD2  $SDIR