if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_v2" ]; then
    mkdir ./logs/LongForecasting_v2
fi
seq_len=96
model_name=PathFormer

root_path_name=./dataset/weather
data_path_name=weather.csv
model_id_name=weather
data_name=custom

for random_seed in 2024
do
for pred_len in 336 
do
    python -u run.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_name'_'$model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --num_nodes 21 \
      --layer_nums 1 \
      --patch_size_list 16 12 8 4 \
      --residual_connection  1\
      --k 2\
      --d_model 8 \
      --d_ff 64 \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 256 --learning_rate 0.001 >logs/LongForecasting_v2/22_$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
done
