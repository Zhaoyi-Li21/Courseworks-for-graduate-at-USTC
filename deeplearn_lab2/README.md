### Sentiment Classification

#### package version -- example
```
python == 3.8.8
pytorch == 1.10.0
torchtext == 0.11.0
# you can refer to this link:
https://blog.csdn.net/meiqi0538/article/details/123459081
https://gitcode.net/mirrors/pytorch/text?utm_source=csdn_github_accelerator
to install the proper versions of torchtext according to your pytorch version.
```

#### unzip the dataset
```
mkdir dataset
cd dataset
#download the dataset package in this directory from the following link:
https://ai.stanford.edu/~amaas/data/sentiment/
tar -zxvf aclImdb_v1.tar.gz
mkdir procd
```
#### preprocess the dataset
```
cd ../src
# follow the `preprocess.ipynb` step-by-step to generate processed .csv file
# generate `train/dev/test.csv` file
```

#### train the Model with the preprocessed dataset
```
# follow the `sent_cls.ipynb` step-by-step
# you can change the hyperparameters with your own customized settings;
TEXT.build_vocab(train, vectors="glove.6B.100d", max_size=10000, min_freq=10)
# you can also download the glove-word2vec-file with the following link:
https://link.zhihu.com/?target=https%3A//apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip

# after training for 25 epochs, testing acc -> 90%
# you can directly enter the sent_cls.ipynb to see the detailed training infos.
```

#### experiment results (hyperparameters)
We search hyperparameters (bidirectional, layer_num, batch_size, hidden_dim)
in the following SearchSpace:
{True, False} $\times$ {1, 2} $\times$ {16, 64} $\times$ {128, 256}
(16 groups of experiment in total)
```
@1:bidirectional = True, layer_num = 2
hyper-parameter setting	testing performance(accuracy)
batch_size=16,hidden_dim=256	89.95%
batch_size=16,hidden_dim=128	90.29%
batch_size=64,hidden_dim=256	89.60%
batch_size=64,hidden_dim=128	90.41%
```
```
@2:bidirectional = True, layer_num = 1
hyper-parameter setting	testing performance(accuracy)
batch_size=16,hidden_dim=256	91.55%
batch_size=16,hidden_dim=128	91.63%
batch_size=64,hidden_dim=256	91.78%
batch_size=64,hidden_dim=128	91.75%
```
```
@3:bidirectional = False, layer_num = 2
hyper-parameter setting	testing performance(accuracy)
batch_size=16,hidden_dim=256	90.14%
batch_size=16,hidden_dim=128	89.44%
batch_size=64,hidden_dim=256	90.28%
batch_size=64,hidden_dim=128	89.97%
```
```
@4:bidirectional = False, layer_num = 1
hyper-parameter setting	testing performance(accuracy)
batch_size=16,hidden_dim=256	91.69%
batch_size=16,hidden_dim=128	91.53%
batch_size=64,hidden_dim=256	91.64%
batch_size=64,hidden_dim=128	91.43%
```