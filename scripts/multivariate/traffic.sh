if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PathFormer

root_path_name=./dataset/traffic
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

for random_seed in 22
do
for pred_len in 720
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
      --patch_size_list 16 12 8 32\
      --num_nodes 862 \
      --layer_nums 1 \
      --k 2\
      --d_model 16 \
      --d_ff 128 \
      --train_epochs 50\
      --residual_connection 1\
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --itr 1 \
      --batch_size 4 --learning_rate 0.0002 >logs/LongForecasting/$random_seed'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done

