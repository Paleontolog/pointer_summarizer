from __future__ import unicode_literals, print_function, division

import time

import torch
import youtokentome
from nltk import word_tokenize
from nltk.stem.snowball import RussianStemmer
from pymystem3 import mystem

from data_util import batcher
from data_util import data, config
from data_util.batcher import Batcher
from data_util.data import *
from data_util.metrics import *
from data_util.preprocess import preprocess_gramm, preprocess_stemm, preprocess_lemm
from training_ptr_gen.model import Model
from training_ptr_gen.train_util import get_input_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path, model_type="stem", load_batcher=True):

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        if load_batcher:
            self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                                   batch_size=config.beam_size, single_pass=True)
            time.sleep(15)
        self.model = Model(model_file_path, is_eval=True)
        self.model_type = model_type

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def restore_text(self, text):
        if self.model_type == "stem":
            return " ".join(text).replace(" +", "")
        elif self.model_type == "gram":
            return "".join(text).replace(" ", "").replace("▁", " ")
        else:
            return " ".join(text)

    def decode(self):
        lemm = pymystem3.Mystem()
        rouge = RougeCalculator(stopwords=True, lang=LangRU())
        result_rouge = [0] * 6

        batch = self.batcher.next_batch()

        iters = 0
        while batch is not None:
            # Run beam search to get best Hypothesis
            with torch.no_grad():
                best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]
            original_text = batch.original_articles
            article_oov = batch.art_oovs[0] if batch.art_oovs else None

            batch = self.batcher.next_batch()

            original_abstract_sents = self.restore_text(original_abstract_sents)
            decoded_words_restore = self.restore_text(decoded_words)
            decoded_words = " ".join(decoded_words)

            print(f"original_abstract : {original_abstract_sents}")
            print(f"original_text : {original_text}")
            print(f"decoded_words : {decoded_words_restore}")
            print(f"decoded_words_oov : {show_abs_oovs(decoded_words, self.vocab, article_oov)}")

            cur_rouge = calk_rouge(original_abstract_sents, [decoded_words_restore], rouge, lemm)

            result_rouge = list(map(lambda x: x[0] + x[1], zip(result_rouge, cur_rouge)))
            iters += 1

        print("--" * 100)
        print("RESULT METRICS")
        result_rouge = [i / iters for i in result_rouge]
        print_results(result_rouge)
        print("++++" * 100)

    def beam_search(self, batch):
        # batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, \
        enc_batch_extend_vocab, extra_zeros, \
        c_t_0, coverage_t_0 = get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h, dec_c = dec_h.squeeze(), dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]

        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]

            y_t_1 = torch.tensor(latest_tokens, dtype=torch.long)
            if use_cuda:
                y_t_1 = y_t_1.cuda()

            all_state_h, all_state_c, all_context = [], [], []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0),
                     torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = [h.coverage for h in beams]
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = \
                self.model.decoder(y_t_1, s_t_1,
                                   encoder_outputs, encoder_feature,
                                   enc_padding_mask, c_t_1,
                                   extra_zeros, enc_batch_extend_vocab,
                                   coverage_t_1, steps)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h, dec_c = dec_h.squeeze(), dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    if h.latest_token != self.vocab.word2id(data.UNKNOWN_TOKEN):
                        beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)
        return beams_sorted[0]

    def test_calc(self, article):
        example = batcher.Example(article, [], self.vocab)
        batch = batcher.Batch([example for _ in range(config.beam_size)],
                              self.vocab, config.beam_size)

        with torch.no_grad():
            best_summary = self.beam_search(batch)

        output_ids = [int(t) for t in best_summary.tokens[1:]]
        decoded_words = data.outputids2words(output_ids, self.vocab,
                                             (batch.art_oovs[0] if config.pointer_gen else None))

        article_restore = self.restore_text(batch.original_articles[-1].split())
        decoded_words_restore = self.restore_text(decoded_words).replace("[STOP]", "")
        print(f"original_text : {article_restore}")
        print(f"decoded_words : {decoded_words_restore}")
        decoded_words = " ".join(decoded_words)
        print(f"decoded_words_oov : {show_abs_oovs(decoded_words, self.vocab, batch.art_oovs[0] if batch.art_oovs else None)}")

    def test(self, mode, bpe_model_path=None):
        while True:
            file_path = input("File path: ").strip()
            file_path = r"C:\Users\lezgy\OneDrive\Рабочий стол\Data_summ\data.txt"
            if file_path == "q":
                break
            try:
                with open(file_path, "r", encoding="utf-8") as r:
                    article = r.read().strip().split("\n")
                    article = " ".join(article)
                    if mode in ["lemm", "stem", "gram", "base"]:
                        article = article.lower()
                        article = word_tokenize(article)
                        article = " ".join(article)
                    print(f"real_text : {article}")

                if mode == "lemm":
                    lemmatizer = mystem.Mystem()
                    article = preprocess_lemm(article, lemmatizer)
                elif mode == "stem":
                    stemmer = RussianStemmer(False)
                    article = preprocess_stemm(article, stemmer)
                elif mode == "gram":
                    token_model = youtokentome.BPE(model=bpe_model_path)
                    article = preprocess_gramm(article, token_model)
                self.test_calc(article)
            except Exception as e:
                print(e)
                print("File not found")


if __name__ == '__main__':
    # "w2v_base": r"D:\PycharmProjects\pointer_summarizer\w2v_model\base\train_1589331529\model\best\model_175000_1589508725",
    models_paths = {
        "fool_model_base": r"D:\PycharmProjects\pointer_summarizer\fool_model\train_1588595388\model\critical\model_200000_1588864682",
        "fool_model_cov": r"D:\PycharmProjects\pointer_summarizer\fool_model\train_1588595388\model\coverage\model_220000_1588873707",
        "w2v_base": r"D:\PycharmProjects\pointer_summarizer\w2v_model\base\train_1589331529\model\best\model_195000_1589512775",
        "w2v_cov": r"D:\PycharmProjects\pointer_summarizer\w2v_model\coverage\coverage\model_176000_1591025363",
        "stem_critical": r"D:\PycharmProjects\pointer_summarizer\log_ria_stem\base\train_1588957995\model\critical\model_160000_1589035328",
        "stem_best": r"D:\PycharmProjects\pointer_summarizer\log_ria_stem\base\train_1588957995\model\last\model_220000_1589064038",
        "stem_cov": r"D:\PycharmProjects\pointer_summarizer\log_ria_stem\coverage\train_1589207752\model\coverage\model_222000_1589208523",
        "gram_best": r"D:\PycharmProjects\pointer_summarizer\log_ria_gramm\new\train_1589144511\model\last\model_65000_1589172657",
        "gram_cov": r"D:\PycharmProjects\pointer_summarizer\log_ria_gramm\new_coverage\train_1589380863\model\coverage\model_64000_1589383826"
    }

    model_filename = models_paths["stem_cov"]
    beam_Search_processor = BeamSearch(model_filename, model_type="stem", load_batcher=False)
    #beam_Search_processor.decode()
    beam_Search_processor.test("stem",
                                bpe_model_path=r"D:\PycharmProjects\pointer_summarizer\tokenize_ria.model")
