import torch
from torch import nn
import torch.nn.functional as F

import onmt
from onmt.translate.beam_search import BeamSearch
from onmt.translate.greedy_search import GreedySearch

def _decode_and_generate(
        model,
        decoder_in,
        memory_bank,
        memory_lengths,
        step=None,
):
    # Decoder forward, takes [tgt_len, batch, nfeats] as input
    # and [src_len, batch, hidden] as memory_bank
    # in case of inference tgt_len = 1, batch = beam times batch_size
    # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
    dec_out, dec_attn = model.decoder(
        decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
    )

    # Generator forward.
    if "std" in dec_attn:
        attn = dec_attn["std"]
    else:
        attn = None
    log_probs = model.generator(dec_out.squeeze(0))
    # returns [(batch_size x beam_size) , vocab ] when 1 step
    # or [ tgt_len, batch_size, vocab ] when full sentence
    return log_probs, attn


def split_src_to_prevent_padding(src, src_lengths):
    min_len_batch = torch.min(src_lengths).item()
    target_prefix = None
    if min_len_batch > 0 and min_len_batch < src.size(0):
        target_prefix = src[min_len_batch:]
        src = src[:min_len_batch]
        src_lengths[:] = min_len_batch
    return src, src_lengths, target_prefix

def tile_to_beam_size_after_initial_step(model, fn_map_state, log_probs):
    if fn_map_state is not None:
        log_probs = fn_map_state(log_probs, dim=1)
        model.decoder.map_state(fn_map_state)
        log_probs = log_probs[-1]
    return log_probs

def _run_encoder(src, src_lengths, model, max_len_src=None):
    batch_size = len(src_lengths)
    enc_states, memory_bank, src_lengths = model.encoder(
        src, src_lengths, max_len_src
    )
    if src_lengths is None:
        assert not isinstance(
            memory_bank, tuple
        ), "Ensemble decoding only supported for text data"
        src_lengths = (
            torch.Tensor(batch_size)
            .type_as(memory_bank)
            .long()
            .fill_(memory_bank.size(0))
        )
    return src, enc_states, memory_bank, src_lengths

def _translate_batch_with_strategy(
    model, src, src_lengths, decode_strategy, max_len_src=None
):
    """Translate a batch of sentences step by step using cache.
    Args:
        batch: a batch of sentences, yield by data iterator.
        src_vocabs (list): list of torchtext.data.Vocab if can_copy.
        decode_strategy (DecodeStrategy): A decode strategy to use for
            generate translation step by step.
    Returns:
        results (dict): The translation results.
    """
    # (0) Prep the components of the search.
    parallel_paths = decode_strategy.parallel_paths  # beam_size

    # (1) Run the encoder on the src.
    src, enc_states, memory_bank, src_lengths = _run_encoder(src, src_lengths, model, max_len_src)
    model.decoder.init_state(src, memory_bank, enc_states)


    # (2) prep decode_strategy. Possibly repeat src objects.
    src_map = None
    target_prefix = None
    (
        fn_map_state,
        memory_bank,
        memory_lengths,
        src_map,
    ) = decode_strategy.initialize(
        memory_bank, src_lengths, src_map, 
        #target_prefix=target_prefix
    )
    if fn_map_state is not None:
        model.decoder.map_state(fn_map_state)

    # (3) Begin decoding step by step:
    for step in range(decode_strategy.max_length):
        decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

        log_probs, attn = _decode_and_generate(
            model,
            decoder_input,
            memory_bank,
            memory_lengths=memory_lengths,
            step=step,
        )

        decode_strategy.advance(log_probs, attn)
        any_finished = decode_strategy.is_finished.any()
        if any_finished:
            decode_strategy.update_finished()
            if decode_strategy.done:
                break

        select_indices = decode_strategy.select_indices

        if any_finished:
            # Reorder states.
            if isinstance(memory_bank, tuple):
                memory_bank = tuple(
                    x.index_select(1, select_indices) for x in memory_bank
                )
            else:
                memory_bank = memory_bank.index_select(1, select_indices)

            memory_lengths = memory_lengths.index_select(0, select_indices)

            if src_map is not None:
                src_map = src_map.index_select(1, select_indices)

        if parallel_paths > 1 or any_finished:
            model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices)
            )

    return decode_strategy.predictions, decode_strategy.scores


def get_onmt_transformer_seq2seq_model(vocab_x, vocab_y, transformer_config):
    if transformer_config == '3layer':
        emb_size = 256
        dropout_rate=0.1
        d_model = 256
        n_head = 4
        d_ff = 512
        n_layers = 3
        max_relative_positions = 20
    elif transformer_config == '6layer':
        emb_size = 256
        dropout_rate=0.1
        d_model = 256
        n_head = 4
        d_ff = 512
        n_layers = 6
        max_relative_positions = 20
    else:
        raise ValueError
    encoder_embeddings = onmt.modules.Embeddings(word_vec_size=emb_size, word_vocab_size=len(vocab_x), word_padding_idx=vocab_x.pad(), position_encoding=True, dropout=dropout_rate )
    encoder = onmt.encoders.TransformerEncoder(num_layers = n_layers,
                                               d_model = d_model,
                                               heads = n_head,
                                               d_ff = d_ff,
                                               dropout = dropout_rate,
                                               attention_dropout = dropout_rate,
                                               embeddings= encoder_embeddings,
                                               max_relative_positions = max_relative_positions)
    decoder_embeddings = onmt.modules.Embeddings(word_vec_size=emb_size, word_vocab_size=len(vocab_y), word_padding_idx=vocab_y.pad(), position_encoding=True, dropout=dropout_rate )
    decoder = onmt.decoders.TransformerDecoder(num_layers= n_layers,
                                               d_model= d_model,
                                               heads= n_head,
                                               d_ff = d_ff,
                                               copy_attn=False,
                                               self_attn_type='scaled-dot',
                                               dropout=dropout_rate,
                                               attention_dropout=dropout_rate,
                                               embeddings=decoder_embeddings,
                                               max_relative_positions=max_relative_positions,
                                               aan_useffn=False,
                                               full_context_alignment=False,
                                               alignment_layer=-3,
                                               alignment_heads=0,
                                               )
    model = onmt.models.model.NMTModel(encoder, decoder)
    return model, d_model, encoder_embeddings, decoder_embeddings

class TransformerEncDec(nn.Module):
    def __init__(self,
                 vocab_x,
                 vocab_y,
                 MAXLEN=45,
                 transformer_config=None
                ):

        super().__init__()

        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        #self.nll = nn.CrossEntropyLoss(ignore_index=vocab_y.pad()) #TODO: why mean better?
        #self.nll = nn.CrossEntropyLoss(ignore_index=vocab_y.pad())
        self.nll = nn.CrossEntropyLoss()
        '''
        note that : I make smoothly train the model only when I do not use the 
        `ignore_index`,
        why: ?
        My analysis: `padding` part is very easy to learn, so when we do not 
        ignore padding, the loss will decrease very fast
        '''
        self.MAXLEN = MAXLEN

        self.model, output_dim, self.encoder_embeddings, self.decoder_embeddings\
            = get_onmt_transformer_seq2seq_model(vocab_x, vocab_y, transformer_config)

        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

        self.model.generator = nn.Sequential(
            nn.Linear(output_dim, len(vocab_y)),
        )

        self.output_dim = output_dim

    def forward(self, inp, out, lens=None, batch_sum=True, test=False):
        #print('inp:', inp.shape, inp[:,1])
        #print('out:', out.shape, out[:,1])
        #print('lens:', lens.shape, lens[1])
        #if test == True:
        #    self.nll = nn.CrossEntropyLoss(ignore_index=self.vocab_y.pad())

        model_outputs = self.model(inp.unsqueeze(-1), out.unsqueeze(-1), lens, self.MAXLEN)
        # I modify the original source code to make it possible to pass my own-set `max_len_src` parameter;

        dec_out = model_outputs[0]
        #print('dec_out:', dec_out.shape, dec_out[:,1])
        logits = self.model.generator(dec_out)
        # print(logits.shape)
        # print('logits:', logits.shape, logits[:,1])
        # logits.shape = [seq_len, bs, vocab_size]
        out_tgt = out[1:, :]
        # out_tgt.shape = [seq_len, bs]
        # print(out_tgt.shape)
        if batch_sum == True:
            seq_len, bs = out_tgt.shape[0], out_tgt.shape[1]
            out_tgt = out_tgt.reshape(seq_len * bs, 1).squeeze(1)
            dec = logits.view(-1, len(self.vocab_y))
            loss = self.nll(dec, out_tgt) / inp.shape[1]
            #loss = self.nll(dec, out_tgt)
        else:
            # loss (for each sample)
            sample_losses = list()
            for i in range(inp.shape[1]):
                sample_pred = logits[: , i , :] # shape = [out_len, label_num]
                sample_label = out_tgt[: , i] # shape = [out_len]
                # we now calculate loss for each single example
                sample_loss = self.nll(sample_pred, sample_label) # shape = []
                '''
                according to MET-PRIM calculation, we do not do reduction
                '''
                sample_losses.append(sample_loss)
            loss = torch.stack(sample_losses) # shape = [bs]
        return loss

    def logprob(self, inp, out, lens=None):
        raise NotImplementedError

    def logprob_interleaved(self, inp, out, lens=None):
        raise NotImplementedError

    def sample(
            self,
            inp,
            max_len,
            lens=None,
            prompt=None,
            greedy=False,
            top_p=None,
            temp=1.0,
            custom_sampler=None,
            beam_size=1,
            calc_score=False,
            n_best=1,
            **kwargs):
        assert(prompt is None)
        assert(lens is not None)

        decode_strategy = self.get_decode_strategy(len(lens), max_len, greedy, top_p, temp, beam_size, n_best)

        sample_results = _translate_batch_with_strategy(self.model, inp.unsqueeze(-1), lens, decode_strategy, self.MAXLEN)


        preds, others = sample_results
        # clean pred results
        out_preds = []
        if len(preds[0]) == 1:
            # only returning single prediction
            for p in preds:
                out_preds.append(p[0].cpu().numpy().tolist())
        else:
            for p in preds:
                out_preds.append([x.cpu().numpy().tolist() for x in p])

        return out_preds, others



    def get_decode_strategy(self, batch_size, max_len, greedy, top_p, temp, beam_size, n_best):
        scorer = onmt.translate.GNMTGlobalScorer(alpha=0.0, beta=0.0, length_penalty='none', coverage_penalty='none')
        if beam_size > 1:
            decode_strategy = BeamSearch(
                beam_size,
                batch_size=batch_size,
                pad=self.vocab_y.pad(),
                bos=self.vocab_y.sos(),
                eos=self.vocab_y.eos(),
                # unk=self.vocab_y.unk(),
                n_best=n_best,
                global_scorer=scorer,
                min_length=0,
                max_length= max_len,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                return_attention=False,
                stepwise_penalty=False,
                ratio=0.0,
                # ban_unk_token=True
            )
        else:

            decode_strategy = GreedySearch(
                pad=self.vocab_y.pad(),
                bos=self.vocab_y.sos(),
                eos=self.vocab_y.eos(),
                # unk=self.vocab_y.unk(),
                batch_size=batch_size,
                # global_scorer=scorer,
                min_length=0,
                max_length= max_len,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                return_attention=False,
                sampling_temp=temp,
                keep_topk=1 if greedy else -1,
                # keep_topp=0.0 if top_p is None else top_p,
                # beam_size = 1,
                # ban_unk_token=True
            )


        return decode_strategy



