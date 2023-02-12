run_test () {
    SOURCE_TOPIC=$1
    TARGET_TOPIC=$2
    LAMBDA=$3
    SAVE_DIR=$4
    EXTRA_OPTIONS=$5

    python train_classic.py \
        --source_topic $SOURCE_TOPIC \
        --target_topic $TARGET_TOPIC \
        --save_dir $SAVE_DIR/body_anon_lower_$SOURCE_TOPIC-$TARGET_TOPIC\[hidden-64\,lmbd-$LAMBDA\,bert-unfrozen] \
        --lmbd $LAMBDA \
        --hidden_dim 64 \
        --num_epochs 5 \
        $EXTRA_OPTIONS \
        --n_runs 5
}

LMBD1="0.5"
LMBD2="1.0"
SDIR="../models/unfrozen_linearlmbd_classic"

mkdir $SDIR

# run_test "politics" "sports" "0.0"  $SDIR
# run_test "politics" "sports" $LMBD1 $SDIR
# # run_test "politics" "sports" $LMBD2 $SDIR

# run_test "politics" "social" "0.0"  $SDIR
# run_test "politics" "social" $LMBD1 $SDIR
# # run_test "politics" "social" $LMBD2 $SDIR

# run_test "social" "sports" "0.0"  $SDIR
# run_test "social" "sports" $LMBD1 $SDIR
# # run_test "social" "sports" $LMBD2 $SDIR

# run_test "social" "politics" "0.0"  $SDIR
# run_test "social" "politics" $LMBD1 $SDIR
# # run_test "social" "politics" $LMBD2 $SDIR

# run_test "sports" "politics" "0.0"  $SDIR
# run_test "sports" "politics" $LMBD1 $SDIR
# # run_test "sports" "politics" $LMBD2 $SDIR

# run_test "sports" "social" "0.0"  $SDIR
# run_test "sports" "social" $LMBD1 $SDIR
# # run_test "sports" "social" $LMBD2 $SDIR

SDIR="../models/unfrozen_linearlmbd_classic_textonly"
mkdir $SDIR

# run_test "politics" "sports" "0.0"  $SDIR "--text_only"
# run_test "politics" "sports" $LMBD1 $SDIR "--text_only"
# # run_test "politics" "sports" $LMBD2 $SDIR "--text_only"

# run_test "social" "politics" "0.0"  $SDIR "--text_only"
# run_test "social" "politics" $LMBD1 $SDIR "--text_only"
# # run_test "social" "politics" $LMBD2 $SDIR "--text_only"

SDIR="../models/unfrozen_linearlmbd_classic_imagesonly"

mkdir $SDIR

# run_test "politics" "sports" "0.0"  $SDIR "--images_only"
# run_test "politics" "sports" $LMBD1 $SDIR "--images_only"
# # run_test "politics" "sports" $LMBD2 $SDIR "--images_only"

# run_test "social" "politics" "0.0"  $SDIR "--images_only"
run_test "social" "politics" $LMBD1 $SDIR "--images_only"
# run_test "social" "politics" $LMBD2 $SDIR "--images_only"