if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PathFormer

root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

for random_seed in 22 123 999 2024
do
for pred_len in 96
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
      --num_nodes 7 \
      --layer_nums 1 \
      --residual_connection 0 \
      --k 2 \
      --d_model 4 \
      --train_epochs 30\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 512 --learning_rate 0.0005 >logs/LongForecasting/$random_seed'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done



for pred_len in 192
do
    python -u run.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
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
      --d_model 8 \
      --train_epochs 30\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 512 --learning_rate 0.0005 >logs/LongForecasting/$random_seed'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


for pred_len in 336
do
    python -u run.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
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
      --batch_size 512 --learning_rate 0.0005 >logs/LongForecasting/$random_seed'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


for pred_len in 720
do
    python -u run.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --patch_size_list 16 12 8 32\
      --num_nodes 7 \
      --layer_nums 1 \
      --residual_connection 0 \
      --k 3 \
      --d_model 16 \
      --train_epochs 30\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 512 --learning_rate 0.0005 >logs/LongForecasting/$random_seed'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

done



