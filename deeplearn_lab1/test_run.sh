CUDA_VISIBLE_DEVICES=9 python train.py \
                                -lr 1e-3 \
                                -actfunc relu \
                                -width 20 \
                                -depth 2 > results/exp.test.txt