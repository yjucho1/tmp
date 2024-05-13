if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=36
model_name=PathFormer

root_path_name=./dataset/illness/
data_path_name=national_illness.csv
model_id_name=illness
data_name=custom

for random_seed in 22 123 999 2024
do
for pred_len in 24 36 48 60
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
      --patch_size_list 6 3 2 12 \
      --num_nodes 7 \
      --layer_nums 1 \
      --k 2\
      --d_model 16 \
      --d_ff 64 \
      --train_epochs 30\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 16 --learning_rate 0.0025 >logs/LongForecasting/$random_seed'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done
