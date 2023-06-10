home='/data2/home/zhaoyi/labs/USTC-labs/natural_language_understanding/machine_translation/transformer'

CUDA_VISIBLE_DEVICES=6 python $home/main.py \
            -gpu_id 8 \
            -data_path $home/data/ch_en_all.txt \
            -epochs 3000 \
            -batch_size 256 \
            -lr 0.01 \
            -model_config '6layer' \
            > $home/results/exp.6.out 2> $home/results/exp.6.err
