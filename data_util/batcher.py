from multiprocessing import Queue, Process
from queue import Empty
import time
from random import shuffle
from threading import Thread
import numpy as np
from data_util import config
from data_util import data
import random
import glob
import multiprocessing

random.seed(1234)

class Example:

    def __init__(self, article, abstract_sentences, vocab):
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        article_words = article.split()

        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words)

        # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input = [vocab.word2id(w) for w in article_words]

        # Process the abstract
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        # list of word ids; OOVs are represented by the id for UNK token
        abs_ids = [vocab.word2id(w) for w in abstract_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids,
                                                                 config.max_dec_steps,
                                                                 start_decoding,
                                                                 stop_decoding)
        self.dec_len = len(self.dec_input)

        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id;
            # also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding,
                                                        stop_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:
            inp = inp[:max_len]
            target = target[:max_len]
        else:
            target.append(stop_id)
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch:
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(data.PAD_TOKEN)
        self.init_encoder_seq(example_list)
        self.init_decoder_seq(example_list)
        self.store_orig_strings(example_list)

    def init_encoder_seq(self, example_list):
        # Maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]
        self.original_abstracts = [ex.original_abstract for ex in example_list]
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]



class FillExampleQueue(multiprocessing.Process):
    def __init__(self, idx, data_path, vocab, thread_num, queue, single_pass=None, min_len=1 ):
        super().__init__()
        self.idx = idx
        self.data_path = data_path
        self.single_pass = single_pass
        self.vocab = vocab
        self.thread_num = thread_num
        self.queue = queue
        self.single_pass = single_pass
        self.alive = True
        self.read_file_list()
        self.min_len = min_len

    def read_file_list(self):
        filelist = glob.glob(self.data_path)
        assert filelist, ('Error: Empty filelist at %s' % self.data_path)
        if self.single_pass:
            filelist = sorted(filelist)
            self.file_list = filelist
        else:
            filelist = sorted(filelist)
            self.size = len(filelist) // self.thread_num
            self.file_list = filelist[self.idx * self.size: (self.idx + 1) * self.size]
            shuffle(self.file_list)
        print(f"Thread: {self.idx}, num chunks: {len(self.file_list)}")

    def read_file(self, file_name):
        with open(file_name, "r", encoding="utf-8") as r:
            file_data = [tuple(line.strip().split("<p>")) for line in r.readlines()]
            shuffle(file_data)
            return file_data

    def generator(self):
        while True:
            for file in self.file_list:
                for line in self.read_file(file):
                    yield line
            if self.single_pass:
                break
            else:
                shuffle(self.file_list)

    def run(self):
        self.example_generator = self.generator()
        while self.alive:
            if not self.queue.full():
                try:
                    abstract, article = next(self.example_generator)
                except StopIteration:
                    self.alive = False
                    break
                except ValueError:
                    continue
                if len(abstract) <= self.min_len or len(article) <= self.min_len:
                    continue
                abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)]
                example = Example(article, abstract_sentences, self.vocab)
                self.queue.put(example)

    def stop(self):
        self.alive = False


class FillBatchQueue(multiprocessing.Process):
    def __init__(self, mode, vocab, example_queue, batch_queue,
                 batch_size, bucketing_cache_size, single_pass):
        super().__init__()
        self.vocab = vocab
        self.mode = mode
        self.example_queue = example_queue
        self.batch_queue = batch_queue
        self.batch_size = batch_size
        self.bucketing_cache_size = bucketing_cache_size
        self.single_pass = single_pass
        self.alive = True

    def stop(self):
        self.alive = False

    def run(self):
        while self.alive:
            if self.mode == 'decode':
                # beam search decode mode single example repeated in the batch
                try:
                    ex = self.example_queue.get(timeout=20 if self.single_pass else None)
                    b = [ex for _ in range(self.batch_size)]
                    self.batch_queue.put(Batch(b, self.vocab, self.batch_size))
                except Empty:
                    self.alive = False
                    break
            else:
                inputs = []
                for _ in range(self.batch_size * self.bucketing_cache_size):
                    try:
                        ex = self.example_queue.get(timeout=20 if self.single_pass else None)
                        inputs.append(ex)
                    except Empty:
                        self.alive = False
                        break

                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True)

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self.single_pass:
                    shuffle(batches)
                for b in batches:
                    if len(b) == self.batch_size:
                        self.batch_queue.put(Batch(b, self.vocab, self.batch_size))


class Batcher(object):
    BATCH_QUEUE_MAX = 100  # max size of batch_queue

    def __init__(self, data_path, vocab, mode, batch_size, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        self._batch_queue = Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        if single_pass:
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1
            self._bucketing_cache_size = 1
        else:
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 2
            self._bucketing_cache_size = 50

        self.start_threads()

        if not single_pass:
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def start_threads(self):
        self._example_q_threads = []
        for idx in range(self._num_example_q_threads):
            self._example_q_threads.append(FillExampleQueue(idx,
                                                            self._data_path,
                                                            self._vocab,
                                                            self._num_example_q_threads,
                                                            self._example_queue,
                                                            self._single_pass))
            if not self._single_pass:
                self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(FillBatchQueue(self.mode,
                                                        self._vocab,
                                                        self._example_queue,
                                                        self._batch_queue,
                                                        self.batch_size,
                                                        self._bucketing_cache_size,
                                                        self._single_pass))
            if not self._single_pass:
                self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

    def next_batch(self):
        if self._batch_queue.qsize() == 0:
            if self._single_pass and (not self._example_q_threads[-1].is_alive()):
                self._example_q_threads[-1].stop()
                self._batch_q_threads[-1].stop()
                return None
        try:
            batch = self._batch_queue.get(timeout=20 if self._single_pass else None)
            return batch
        except Empty:
            return None

    def watch_threads(self):
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():
                    new_t = FillExampleQueue(idx,
                                             self._data_path,
                                             self._vocab,
                                             self._num_example_q_threads,
                                             self._example_queue,
                                             self._single_pass)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():
                    new_t = FillBatchQueue(self.mode,
                                           self._vocab,
                                           self._example_queue,
                                           self._batch_queue,
                                           self.batch_size,
                                           self._bucketing_cache_size,
                                           self._single_pass)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
