export XRT_TPU_CONFIG="localservice;0;localhost:51011"

# ############ Baseline T3 Directly Trained #################
# export OUTPUT_DIR=conll-model-t3-4ep
# export BATCH_SIZE=32
# export NUM_EPOCHS=4.
# export SAVE_STEPS=750
# export SEED=42
# export MAX_LENGTH=128
# export BERT_MODEL=bert-base-cased
# python3 -u cornell-train.py \
# --data_dir conll2003 \
# --model_type bert \
# --model_name_or_path $BERT_MODEL \
# --output_dir $OUTPUT_DIR \
# --max_seq_length  $MAX_LENGTH \
# --num_train_epochs $NUM_EPOCHS \
# --per_gpu_train_batch_size $BATCH_SIZE \
# --per_gpu_eval_batch_size $BATCH_SIZE \
# --save_steps $SAVE_STEPS \
# --seed $SEED \
# --nprocs 1 \
# --weight_decay 0.01 \
# --num_hidden_layers 2 \
# --do_train \
# --do_eval \
# --do_predict

# ########### Baseline BERT finetuned on CoNLL #################
# export OUTPUT_DIR=conll-model-base-4ep
# export BATCH_SIZE=32
# export NUM_EPOCHS=4.
# export SAVE_STEPS=750
# export SEED=42
# export MAX_LENGTH=128
# export BERT_MODEL=bert-base-cased
# python3 -u cornell-train.py \
# --data_dir conll2003 \
# --model_type bert \
# --model_name_or_path $BERT_MODEL \
# --output_dir $OUTPUT_DIR \
# --max_seq_length  $MAX_LENGTH \
# --num_train_epochs $NUM_EPOCHS \
# --per_gpu_train_batch_size $BATCH_SIZE \
# --per_gpu_eval_batch_size $BATCH_SIZE \
# --save_steps $SAVE_STEPS \
# --seed $SEED \
# --nprocs 1 \
# --weight_decay 0.01 \
# --do_train \
# --do_eval \
# --do_predict

export OUTPUT_DIR=conll-model-T3-debug5-non-perm
export BATCH_SIZE=32
export NUM_EPOCHS=10.
export SAVE_STEPS=750
export SEED=42
export MAX_LENGTH=128
export BERT_MODEL_TEACHER=conll-model-base-4ep
python3 cornell-distill.py \
--temperature 4 \
--data_dir conll2003 \
--model_type bert \
--model_name_or_path $BERT_MODEL_TEACHER \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--num_hidden_layers 3 \
--save_steps $SAVE_STEPS \
--seed $SEED \
--nprocs 1 \
--weight_decay 0.01 \
--permute_logits 0 \
--do_distill \
--do_train \
--do_eval \
--do_predict

# export OUTPUT_DIR=conll-model-T3-debug5-perm
# export BATCH_SIZE=32
# export NUM_EPOCHS=10.
# export SAVE_STEPS=750
# export SEED=42
# export MAX_LENGTH=128
# export BERT_MODEL_TEACHER=conll-model-base-4ep
# python3 cornell-distill.py \
# --temperature 4 \
# --data_dir conll2003 \
# --model_type bert \
# --model_name_or_path $BERT_MODEL_TEACHER \
# --output_dir $OUTPUT_DIR \
# --max_seq_length  $MAX_LENGTH \
# --num_train_epochs $NUM_EPOCHS \
# --per_gpu_train_batch_size $BATCH_SIZE \
# --num_hidden_layers 3 \
# --save_steps $SAVE_STEPS \
# --seed $SEED \
# --nprocs 1 \
# --weight_decay 0.01 \
# --permute_logits 1 \
# --do_distill \
# --do_train \
# --do_eval \
# --do_predict
