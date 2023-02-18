home=/data2/home/zhaoyi/labs/USTC-labs/deeplearn_lab4_gcn/src
epochs=200
for dataset in 'cora' 'citeseer'
do
    for task in 'linkpred' 'nodecls'
    do
        for pair_norm in True False
        do
            for layer_num in 2 3 4
            do
                for activate in 'relu' 'sigmoid' 'tanh'
                do
                    CUDA_VISIBLE_DEVICES=3 python $home/train.py \
                                                    --dataset $dataset  \
                                                    --pair_norm $pair_norm --activate $activate\
                                                    --epochs $epochs --task $task\
                                                    --layer_num $layer_num 
                done
            done
        done
    done
done
