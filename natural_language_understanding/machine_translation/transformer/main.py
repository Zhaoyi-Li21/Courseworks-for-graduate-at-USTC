from transformer import TransformerEncDec
import torch
from torch import nn, optim
from absl import app, flags, logging
from dataset import Translation_Dataset
import os
import random
import numpy as np
import hlog
from utils import NoamLR
from torch.nn.utils.clip_grad import clip_grad_norm_
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import evaluate


FLAGS = flags.FLAGS
flags.DEFINE_string("model_config", "3layer", "config for the transformer model")
flags.DEFINE_integer("epochs", 1, "how many epochs to be trained")
flags.DEFINE_integer("gpu_id", 0, "which gpu to be used")
flags.DEFINE_string("data_path", None, "the file path of dataset")
flags.DEFINE_integer("seed", 0, "seed for all of the random values")
flags.DEFINE_integer("beam_size", 5, "size for the beam search")
flags.DEFINE_integer("lr_warmup_steps", 4000,"noam warmup_steps")
flags.DEFINE_integer("dim", 512, 'transformer dimension')
flags.DEFINE_float("lr", 0.00001, "learning rate")
flags.DEFINE_integer("batch_size", 64, "how many sentences in a batch")
flags.DEFINE_float("clip", 1. ,"clip gradient")
flags.DEFINE_string("device", "cuda:0", "device to run on")


smooth = SmoothingFunction()
sacrebleu = evaluate.load("sacrebleu")

def main(argv):
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
    device = torch.device(FLAGS.device)

    # set random seed
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # hlog
    hlog.flags()

    # prepare dataloader
    dataset = Translation_Dataset(FLAGS.data_path)
    dataset.trans_lines2data()
    dataset.split_train_dev_test()
    dataset.padding()
    dataset.make_dataloader() # usage: self.dataloader["train"]/["test"]
    dataset.prepare_references_for_bleu()

    # print(dataset.src_vocab["'"])
    # print(dataset.src_vocab["\'"])
    # print(dataset.refs['" Will he pass the examination ? "   " I am afraid not . "'])
    # print(dataset.refs['Wait !'])
    # define model, optimizer and scheduler
    model = TransformerEncDec(dataset.src_vocab, dataset.tgt_vocab, dataset.max_len_src, FLAGS.model_config).to(device)
    opt = optim.Adam(model.parameters(), lr=FLAGS.lr, betas=(0.9, 0.998))
    sched = NoamLR(opt, FLAGS.dim, warmup_steps=FLAGS.lr_warmup_steps)

    # training
    step_cnt = 0
    for i in range(FLAGS.epochs):
        current_loss = 0

        for step, batch in enumerate(dataset.dataloader["train"]):
            
            opt.zero_grad() 
            model.train()

            b_srcs = torch.transpose(batch[0].to(device), 1 ,0) # shape = [max_len_src, bs]
            b_tgts = torch.transpose(batch[1].to(device), 1, 0) # shape = [max_len_tgt, bs]
            b_lens = batch[2].to(device) # shape = [bs]

            # print(b_srcs.shape, b_tgts.shape, b_lens.shape)
            # print(b_srcs[:, 0])
            # print(b_tgts[:, 0])
            # print(b_lens)

            loss = model.forward(b_srcs, b_tgts, b_lens)
            loss.backward()
            gnorm = clip_grad_norm_(model.parameters(), FLAGS.clip)
            if not np.isfinite(gnorm):
                raise Exception("=====GOT NAN=====")
            opt.step()
            sched.step()
            current_loss += loss.item()
            opt.zero_grad() 

            if step_cnt % 100 == 0 and step_cnt != 0:
                # print current_loss
                hlog.value("step", step_cnt)
                hlog.value("training loss", current_loss / 100)
                current_loss = 0

            if step_cnt % 500 == 0 and step_cnt != 0:
                # evaluating use dev set 
                dev_loss = 0
                dev_step_cnt = 0
                
                dev_bleu = 0
                dev_sacrebleu = 0
                with torch.no_grad():
                    model.eval()
                    for _, batch in enumerate(dataset.dataloader["dev"]):
                        batch_bleu = 0
                        batch_sacre_bleu = 0
                        dev_step_cnt += 1
                        b_srcs = torch.transpose(batch[0].to(device), 1 ,0) # shape = [max_len_src, bs]
                        b_tgts = torch.transpose(batch[1].to(device), 1, 0) # shape = [max_len_tgt, bs]
                        _b_srcs = batch[0]
                        #b_tgts = batch[1] # shape = [bs, max_len_tgt]
                        b_lens = batch[2].to(device) # shape = [bs]
                        loss = model.forward(b_srcs, b_tgts, b_lens)
                        dev_loss += loss.item()
                        preds, _ = model.sample(inp=b_srcs, lens=b_lens,
                                    temp=1.0, max_len=dataset.max_len_tgt,
                                    beam_size=FLAGS.beam_size, # in align with previous works, stop beam_size
                                    calc_score=False
                                    )
                        for i in range(len(preds)):
                            batch_bleu += dataset.bleu_score(preds[i], _b_srcs[i].tolist(), smooth)
                            batch_sacre_bleu += dataset.sacrebleu_score(preds[i], _b_srcs[i].tolist(), sacrebleu)
                        dev_bleu += batch_bleu/len(preds)
                        dev_sacrebleu += batch_sacre_bleu/len(preds)
  
                hlog.value("step", step_cnt)
                hlog.value("dev loss", dev_loss / dev_step_cnt)
                hlog.value("dev bleu", dev_bleu / dev_step_cnt)
                hlog.value("dev sacrebleu", dev_sacrebleu/ dev_step_cnt)
                pass

                # evaluating
                if step_cnt % 10000 == 0 and step_cnt != 0:
                    print("SHOWING TRANSLATION RESULT AT STEP:", step_cnt)
                test_loss = 0
                test_step_cnt = 0
                            
                test_bleu = 0
                test_sacrebleu = 0
                with torch.no_grad():
                    model.eval()
                    for _, batch in enumerate(dataset.dataloader["test"]):
                        batch_bleu = 0
                        batch_sacre_bleu = 0
                        test_step_cnt += 1
                        b_srcs = torch.transpose(batch[0].to(device), 1 ,0) # shape = [max_len_src, bs]
                        b_tgts = torch.transpose(batch[1].to(device), 1, 0) # shape = [max_len_tgt, bs]
                        _b_srcs = batch[0]
                        #b_tgts = batch[1] # shape = [bs, max_len_tgt]
                        b_lens = batch[2].to(device) # shape = [bs]
                        loss = model.forward(b_srcs, b_tgts, b_lens)
                        test_loss += loss.item()
                        preds, _ = model.sample(inp=b_srcs, lens=b_lens,
                                                temp=1.0, max_len=dataset.max_len_tgt,
                                                beam_size=FLAGS.beam_size, # in align with previous works, stop beam_size
                                                calc_score=False
                                                )
                        for i in range(len(preds)):
                            '''
                            if step_cnt >= 5000:
                                batch_bleu += dataset.bleu_score(preds[i], _b_srcs[i].tolist(), smooth, True)
                            '''
                            if step_cnt % 10000 == 0 and step_cnt != 0:
                                if random.random() < 0.01:
                                    dataset.show_translation(preds[i], _b_srcs[i].tolist())

                            batch_bleu += dataset.bleu_score(preds[i], _b_srcs[i].tolist(), smooth)
                            batch_sacre_bleu += dataset.sacrebleu_score(preds[i], _b_srcs[i].tolist(), sacrebleu)
                        test_bleu += batch_bleu/len(preds)
                        test_sacrebleu += batch_sacre_bleu/len(preds)

                hlog.value("step", step_cnt)
                hlog.value("test loss", test_loss / test_step_cnt)
                hlog.value("test bleu", test_bleu / test_step_cnt)
                hlog.value("test sacrebleu", test_sacrebleu / test_step_cnt)
            step_cnt += 1    


    # evaluating
    test_loss = 0
    test_step_cnt = 0
                
    test_bleu = 0
    test_sacrebleu = 0
    with torch.no_grad():
        model.eval()
        for _, batch in enumerate(dataset.dataloader["test"]):
            batch_bleu = 0
            batch_sacre_bleu
            test_step_cnt += 1
            b_srcs = torch.transpose(batch[0].to(device), 1 ,0) # shape = [max_len_src, bs]
            b_tgts = torch.transpose(batch[1].to(device), 1, 0) # shape = [max_len_tgt, bs]
            _b_srcs = batch[0]
            #b_tgts = batch[1] # shape = [bs, max_len_tgt]
            b_lens = batch[2].to(device) # shape = [bs]
            loss = model.forward(b_srcs, b_tgts, b_lens)
            test_loss += loss.item()
            preds, _ = model.sample(inp=b_srcs, lens=b_lens,
                                    temp=1.0, max_len=dataset.max_len_tgt,
                                    beam_size=FLAGS.beam_size, # in align with previous works, stop beam_size
                                    calc_score=False
                                    )
            for i in range(len(preds)):
                batch_bleu += dataset.bleu_score(preds[i], _b_srcs[i].tolist(), smooth)
                batch_sacre_bleu += dataset.sacrebleu_score(preds[i], _b_srcs[i].tolist(), sacrebleu)
            test_bleu += batch_bleu/len(preds)
            test_sacrebleu += batch_sacre_bleu/len(preds)
    
    hlog.value("test loss", test_loss / test_step_cnt)
    hlog.value("test bleu", test_bleu / test_step_cnt)
    hlog.value("test scarebleu", test_sacrebleu / test_step_cnt)
    pass



if __name__ == "__main__":

    app.run(main)