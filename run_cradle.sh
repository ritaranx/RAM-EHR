task=cradle
gpu=$1
src=gpt_sum_all
for seed in 0 ; do
for train_seed in 0  ; do 
for model_type in distill-clinicalbert ; do 
seed=${seed}
train_seed=${train_seed}
max_seq_len=400 # 500
max_seq_len_test=400 #500

eval_batch_size=256
steps=100
#####################
# gen_model=gpt3
gpt_model="gpt-3.5-turbo"
data_dir="../${task}"
method_name=${task}-retrieve-def-${src}
train_file="hyperedges-cradle-text-train.jsonl"
valid_file="hyperedges-cradle-text-valid.jsonl"
test_file="hyperedges-cradle-text-test.jsonl"
lr=4e-5 # 2e-5
batch_size=32 # 42
epochs=4
weight_decay=1e-3
use_def=1

output_dir=${task}/model
mkdir -p ${output_dir}
mkdir -p ${task}/cache

train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py --task=${task} \
	--train_file=${train_file} --dev_file=${valid_file} --test_file=${test_file} \
	--unlabel_file=unlabeled.json --tokenizer=${model_type} \
	--gen_model=${gen_model} --data_dir=${data_dir} --seed=${seed} --train_seed=${train_seed} \
	--cache_dir="${task}/cache" --output_dir=${output_dir}  \
	--gpu=${gpu} --num_train_epochs=${epochs} --weight_decay=${weight_decay} --learning_rate=${lr}  \
	--batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
	--max_seq_len=${max_seq_len} --max_seq_len_test=${max_seq_len_test} --auto_load=1 \
	--max_steps=${steps} --model_type=${model_type} --multi_label=1 --use_def=${use_def} --method_name=${method_name}"
echo $train_cmd 
eval $train_cmd

done
done

done
