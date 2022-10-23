lr=1e-3
actfunc='relu'
width=13
depth=3
for lr in 1e-2 5e-3 1e-3 5e-4
do
    for actfunc in 'relu' 'tanh' 'sigmoid'
    do
        width=20
        depth=2
        CUDA_VISIBLE_DEVICES=9 python train.py \
                                -lr $lr \
                                -actfunc $actfunc \
                                -width $width \
                                -depth $depth > results/exp.$lr.$actfunc.$width.$depth.txt
        width=13
        depth=3
        CUDA_VISIBLE_DEVICES=9 python train.py \
                                -lr $lr \
                                -actfunc $actfunc \
                                -width $width \
                                -depth $depth > results/exp.$lr.$actfunc.$width.$depth.txt
        width=10
        depth=4
        CUDA_VISIBLE_DEVICES=9 python train.py \
                                -lr $lr \
                                -actfunc $actfunc \
                                -width $width \
                                -depth $depth > results/exp.$lr.$actfunc.$width.$depth.txt
    done
done