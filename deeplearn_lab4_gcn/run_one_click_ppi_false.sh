home=/data2/home/zhaoyi/labs/USTC-labs/deeplearn_lab4_gcn/src
epochs=200
for dataset in 'ppi'
do
    for task in 'linkpred' 'nodecls'
    do  
        for layer_num in 2 3 4
        do
                for activate in 'relu'
                do
                    for hidden in 16 256
                        do
                            CUDA_VISIBLE_DEVICES=7 python $home/train.py \
                                                    --dataset $dataset  --hidden $hidden\
                                                    --self_loop True --activate $activate\
                                                    --epochs $epochs --task $task\
                                                    --layer_num $layer_num 
                        done
                done
        done
    done
done