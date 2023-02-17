#### References
```
@1: Mu Li's tutorial:
https://www.bilibili.com/video/BV1iT4y1d7zP/?spm_id_from=333.337.search-card.all.click&vd_source=0b94491685a644f4e70b2ffc09079337
@2: Google Research's distill blog:
https://distill.pub/2021/gnn-intro/
@3: pygcn tutorial:
https://www.bilibili.com/video/BV1Y64y1B7Qc/?spm_id_from=333.337.search-card.all.click&vd_source=0b94491685a644f4e70b2ffc09079337
@4: pygcn github (official implementation of GCN in pytorch):
https://github.com/tkipf/pygcn
@5: GCN original paper: (Semi-Supervised Classification with Graph Convolutional Networks, ICLR'17, Thomas N.Kipf, Max Welling)
https://arxiv.org/abs/1609.02907
@6: a blog around GCN:
https://ai.plainenglish.io/graph-convolutional-networks-gcn-baf337d5cb6b?gi=a61c544a76c5
@7: how to use GCN to deal with link prediction task?
https://blog.csdn.net/Cyril_KI/article/details/125956540
@8: EdgeDrop paper:
@9: PairNorm paper:
@10: an example of processing PPI dataset:
https://blog.csdn.net/KPer_Yang/article/details/128810698?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-128810698-blog-112979175.pc_relevant_multi_platform_whitelistv3&spm=1001.2101.3001.4242.1&utm_relevant_index=3
```
CUDA_VISIBLE_DEVICES=1 python train.py --dataset citeseer  --self_loop True --epochs 300 --layer_num 4 --pair_norm True --activate sigmoid

CUDA_VISIBLE_DEVICES=1 python train.py --dataset citeseer  --self_loop True --epochs 300 --layer_num 4 --pair_norm True --activate sigmoid

CUDA_VISIBLE_DEVICES=1 python train.py --dataset citeseer  --self_loop True --epochs 300 --layer_num 2 --pair_norm True --activate relu --task linkpred

CUDA_VISIBLE_DEVICES=1 python train.py --dataset citeseer  --self_loop True --epochs 300 --layer_num 2 --activate relu --task linkpred