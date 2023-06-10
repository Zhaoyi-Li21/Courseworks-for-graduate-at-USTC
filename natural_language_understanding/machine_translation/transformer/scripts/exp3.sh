home='/data2/home/zhaoyi/labs/USTC-labs/natural_language_understanding/machine_translation/transformer'

python $home/main.py \
    -gpu_id 7 \
    -data_path $home/data/ch_en_all.txt \
    -epochs 3000 \
    -batch_size 256 \
    -lr 0.01 \
    -model_config '3layer' \
    > $home/results/exp.0.out 2> $home/results/exp.0.err
