home=/data2/home/zhaoyi/labs/USTC-labs/deeplearn_lab4_gcn/src
epochs=200
for dataset in 'cora' 'citeseer'
do
    for task in 'linkpred' 'nodecls'
    do
        for self_loop in True False
        do
            for layer_num in 2 3 4
            do
                for pair_norm in True False
                do
                    for activate in 'relu' 'sigmoid' 'tanh'
                    do
                        CUDA_VISIBLE_DEVICES=1 python $home/train.py \
                                                    --dataset $dataset  \
                                                    --self_loop $self_loop --activate $activate\
                                                    --epochs $epochs --task $task\
                                                    --layer_num $layer_num --pair_norm $pair_norm 
                    done
                done
            done
        done
    done
done
