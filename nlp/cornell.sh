export XRT_TPU_CONFIG="localservice;0;localhost:51011"

############ Baseline T3 Directly Trained #################
#export OUTPUT_DIR=conll-t3-train-outputs-model-base-3ep
#export BATCH_SIZE=32
#export NUM_EPOCHS=3.
#export SAVE_STEPS=750
#export SEED=42
#export MAX_LENGTH=128
#export BERT_MODEL=bert-base-cased
#python3 -u cornell-train.py \
#--data_dir conll2003 \
#--model_type bert \
#--model_name_or_path $BERT_MODEL \
#--output_dir $OUTPUT_DIR \
#--max_seq_length  $MAX_LENGTH \
#--num_train_epochs $NUM_EPOCHS \
#--per_gpu_train_batch_size $BATCH_SIZE \
#--per_gpu_eval_batch_size $BATCH_SIZE \
#--save_steps $SAVE_STEPS \
#--seed $SEED \
#--num_hidden_layers 3 \
#--do_train \
#--do_eval \
#--do_predict

#export OUTPUT_DIR=conll-outputs-model-base-3ep
#export BATCH_SIZE=32
#export NUM_EPOCHS=3.
#export SAVE_STEPS=750
#export SEED=42
#export MAX_LENGTH=128
#export BERT_MODEL=bert-base-cased
#python3 -u cornell-train.py \
#--data_dir conll2003 \
#--model_type bert \
#--model_name_or_path $BERT_MODEL \
#--output_dir $OUTPUT_DIR \
#--max_seq_length  $MAX_LENGTH \
#--num_train_epochs $NUM_EPOCHS \
#--per_gpu_train_batch_size $BATCH_SIZE \
#--per_gpu_eval_batch_size $BATCH_SIZE \
#--save_steps $SAVE_STEPS \
#--seed $SEED \
#--do_train \
#--do_eval \
#--do_predict

export OUTPUT_DIR=conll-output-model-T3-3ep-test3-debug
export BATCH_SIZE=32
export NUM_EPOCHS=3.
export SAVE_STEPS=750
export SEED=42
export MAX_LENGTH=128
export BERT_MODEL_TEACHER=conll-outputs-model-base-3ep
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
 --do_distill \
 --do_train \
 --do_eval \
 --do_predict
