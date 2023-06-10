home='/data2/home/zhaoyi/labs/USTC-labs/natural_language_understanding/machine_translation/transformer'

python $home/main.py \
    -gpu_id 8 \
    -data_path $home/data/ch_en_all.txt \
    -epochs 200 \
    -batch_size 256 \
    -lr 0.01 \
    -model_config '6layer' \
    > $home/results/test.out 2> $home/results/test.err