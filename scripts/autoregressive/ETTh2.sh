if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Autoregressive" ]; then
    mkdir ./logs/Autoregressive
fi
seq_len=96
model_name=PathFormer

root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

for random_seed in 22 #123 456 99 999
do
for pred_len in 96 192 336 720
do
    python -u run2.py \
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
      --num_nodes 7 \
      --layer_nums 1 \
      --residual_connection 0 \
      --k 2 \
      --d_model 4 \
      --train_epochs 30\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 512 --learning_rate 0.0005 >logs/Autoregressive/$random_seed'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done



