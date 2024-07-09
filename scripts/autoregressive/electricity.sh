if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Autoregressive" ]; then
    mkdir ./logs/Autoregressive
fi
seq_len=96
model_name=PathFormer

root_path_name=./dataset/electricity
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom

for random_seed in 2024
do
for pred_len in 96 192 336 720
do
    python -u run.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --num_nodes 321 \
      --layer_nums 1 \
      --residual_connection 1\
      --k 2\
      --d_model 16 \
      --d_ff 128 \
      --patch_size_list 16 12 8 32 \
      --train_epochs 50\
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --itr 1 \
      --batch_size 8 --learning_rate 0.001 >logs/Autoregressive/$random_seed'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done

