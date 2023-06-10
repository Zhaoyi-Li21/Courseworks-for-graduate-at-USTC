from vocab import Vocab
import random
from absl import flags
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import evaluate

FLAGS = flags.FLAGS

class Translation_Dataset():
    def __init__(self, fr_path, max_src_len=50, max_tgt_len=60):
        fr = open(fr_path,"r")
        self.lines = fr.readlines()
        self.src_vocab = Vocab()
        self.tgt_vocab = Vocab()

    def trans_lines2data(self):
        self.src_strs = list()
        self.tgt_strs = list()

        self.src_data = list()
        self.tgt_data = list()
        self.max_len_src = 0
        self.max_len_tgt = 0

        lines = self.lines
        for line in lines:
            line = line.strip().split('	') # line = [src_str, tgt_str]

            src_str, tgt_str = line[0].split(), line[1].split()

            self.src_strs.append(' '.join(src_str))
            self.tgt_strs.append(' '.join(tgt_str))

            # update vocab
            for tok in src_str:
                self.src_vocab.add(tok)
            for tok in tgt_str:
                self.tgt_vocab.add(tok)
            
            src_datum = self.src_vocab.encode(src_str)
            tgt_datum = self.tgt_vocab.encode(tgt_str)
            if len(src_datum) > self.max_len_src: self.max_len_src = len(src_datum)
            if len(tgt_datum) > self.max_len_tgt: self.max_len_tgt = len(tgt_datum)
            
            self.src_data.append(src_datum)
            self.tgt_data.append(tgt_datum)

    def split_train_dev_test(self, train=0.7, dev=0.15, test=0.15):
        random.seed(FLAGS.seed)
        assert len(self.src_data) == len(self.tgt_data) 
        assert len(self.src_data) > 0
        tot_num = len(self.src_data)
        
        self.set = dict()
        self.set["train"] = list()
        self.set["dev"] = list()
        self.set["test"] = list()

        # split the training set into train // dev // test sets
        for i in range(tot_num):
            datum = (self.src_data[i], self.tgt_data[i])
            rv = random.random()
            if rv <= train:
                self.set["train"].append(datum)
            elif rv <= train + dev:
                self.set["dev"].append(datum)
            else:
                self.set["test"].append(datum)
    
    
    def padding(self):
        self.data = dict()

        for split in ["train", "dev", "test"]:
            self.data[split] = dict()
            self.data[split]["len"] = list()
            self.data[split]["src"] = list()
            self.data[split]["tgt"] = list()

            for datum in self.set[split]:
                src_datum, tgt_datum = datum
                self.data[split]["len"].append(len(src_datum))

                while len(src_datum) < self.max_len_src:
                    src_datum.append(self.src_vocab.pad())

                while len(tgt_datum) < self.max_len_tgt:
                    tgt_datum.append(self.tgt_vocab.pad())

                self.data[split]["src"].append(src_datum)
                self.data[split]["tgt"].append(tgt_datum)


    def make_dataloader(self):
        self.dataset = dict()
        self.dataloader = dict()
        self.sampler = dict()

        for split in ["train", "dev", "test"]:
            # print(len(self.data[split]["src"]))
            # print(len(self.data[split]["tgt"]))
            # print(len(self.data[split]["len"]))
            self.dataset[split] = TensorDataset(torch.tensor(self.data[split]["src"]),
                                                torch.tensor(self.data[split]["tgt"]),
                                                torch.tensor(self.data[split]["len"]))
            self.sampler[split] = RandomSampler(self.dataset[split])
            self.dataloader[split] = DataLoader(self.dataset[split], 
                                                sampler=self.sampler[split], 
                                                batch_size=FLAGS.batch_size)
            # print(FLAGS.batch_size)
            
        pass

    def encode_src(self, inp):
        return self.src_vocab.encode(inp)
    
    def decode_src(self, inp):
        return self.src_vocab.decode(inp)

    def encode_tgt(self, inp):
        return self.tgt_vocab.encode(inp)
    
    def decode_tgt(self, inp):
        return self.tgt_vocab.decode(inp)
    
    def prepare_references_for_bleu(self):
        self.refs = dict()
        for i in range(len(self.src_strs)):
            if self.src_strs[i] not in self.refs.keys():
                self.refs[self.src_strs[i]] = set()
                self.refs[self.src_strs[i]].add(self.tgt_strs[i])
        pass

    def get_refs(self, src_str):
        return list(self.refs[src_str])
        pass

    def bleu_score(self, pred, src, smooth, flag=False):
        
        cand = self.decode_tgt(pred)
        candidates = [' '.join(cand)]
        src = self.decode_src(src)
        src_str = ' '.join(src)
        references = self.get_refs(src_str)
        if flag == True:
            print(candidates, references)
        score = corpus_bleu(references, candidates, smoothing_function=smooth.method1)
        return score
    
    def sacrebleu_score(self, pred, src, sacrebleu):
        cand = self.decode_tgt(pred)
        candidates = [' '.join(cand)]
        src = self.decode_src(src)
        src_str = ' '.join(src)
        references = self.get_refs(src_str)
        results = sacrebleu.compute(predictions=candidates, references=references)
        return round(results["score"], 1)
    
    def show_translation(self, pred, src):
        cand = self.decode_tgt(pred)
        candidates = ' '.join(cand)
        src = self.decode_src(src)
        src_str = ' '.join(src)
        references = self.get_refs(src_str)
        print('---SRC---:', src_str)
        print('---REF---:', references)
        print('---HYPOTHESIS---:', candidates)
    
            
            



     


    
