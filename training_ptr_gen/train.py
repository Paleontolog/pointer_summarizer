import os
import time

import torch
import torch.utils.tensorboard.writer as writer
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_and_write_running_avg_loss, calc_running_avg_loss, write_summary
from training_ptr_gen.checkpoints import Checkpoint
from training_ptr_gen.model import Model
from training_ptr_gen.train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self, train_dir=None, eval_dir=None, vocab=None, vectors=None):
        self.vectors = vectors
        if vocab is None:
            self.vocab = Vocab(config.vocab_path, config.vocab_size)
        else:
            self.vocab = vocab

        print(self.vocab)
        self.batcher_train = Batcher(config.train_data_path, self.vocab, mode='train',
                                     batch_size=config.batch_size, single_pass=False)
        time.sleep(15)
        self.batcher_eval = Batcher(config.eval_data_path, self.vocab, mode='eval',
                                    batch_size=config.batch_size, single_pass=True)
        time.sleep(15)

        cur_time = int(time.time())
        if train_dir is None:
            train_dir = os.path.join(config.log_root, 'train_%d' % (cur_time))
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)

        if eval_dir is None:
            eval_dir = os.path.join(config.log_root, 'eval_%s' % (cur_time))
            if not os.path.exists(eval_dir):
                os.mkdir(eval_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer_train = writer.FileWriter(train_dir)
        self.summary_writer_eval = writer.FileWriter(eval_dir)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path, vectors=self.vectors)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())

        pytorch_total_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"Parameters count: {pytorch_total_params}")

        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        # self.optimizer = adagrad.Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        self.optimizer = Adam(params, lr=initial_lr)
        start_iter, start_training_loss, start_eval_loss = 0, 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_training_loss = state['current_train_loss']
            start_eval_loss = state['current_eval_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            print(k)
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()

        self.chechpoint = Checkpoint(self.model,
                                     self.optimizer,
                                     self.model_dir,
                                     start_eval_loss if start_eval_loss != 0 else float("inf"))

        return start_iter, start_training_loss, start_eval_loss

    def model_batch_step(self, batch, eval):

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        step_decoded_idx = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing

            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = \
                self.model.decoder(y_t_1, s_t_1,
                                   encoder_outputs,
                                   encoder_feature,
                                   enc_padding_mask, c_t_1,
                                   extra_zeros,
                                   enc_batch_extend_vocab,
                                   coverage, di)

            if eval:
                _, top_idx = final_dist.topk(1)
                step_decoded_idx.append(top_idx)

            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        final_decoded_sentences = None
        if eval:
            final_decoded_sentences = torch.stack(step_decoded_idx, 2).squeeze(1)
            print(final_decoded_sentences)

        return loss, final_decoded_sentences

    def train_one_batch(self, batch):
        self.optimizer.zero_grad()
        loss, _ = self.model_batch_step(batch, False)
        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def run_eval(self):

        self.model.eval()
        batch = self.batcher_eval.next_batch()
        iter = 0
        start = time.time()
        running_avg_loss = 0
        with torch.no_grad():
            while batch is not None:
                loss, _ = self.model_batch_step(batch, False)
                loss = loss.item()
                running_avg_loss = calc_running_avg_loss(loss, running_avg_loss)
                batch = self.batcher_eval.next_batch()

                iter += 1
                if iter % config.print_interval == 0:
                    print('Eval steps %d, seconds for %d batch: %.2f , loss: %f' % (
                        iter, config.print_interval, time.time() - start, running_avg_loss))
                    start = time.time()

        return running_avg_loss

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss_train, running_avg_loss_eval = self.setup_train(model_file_path)
        start = time.time()

        loss_train = 0
        while iter < n_iters:

            self.model.train()
            batch = self.batcher_train.next_batch()
            loss_train = self.train_one_batch(batch)
            running_avg_loss_train = calc_and_write_running_avg_loss(loss_train,
                                                                     "running_avg_loss_train",
                                                                     running_avg_loss_train,
                                                                     self.summary_writer_train,
                                                                     iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer_train.flush()

            if iter % config.print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f, loss: %f, avg_loss: %f' % (iter, config.print_interval,
                                                                                        time.time() - start,
                                                                                        loss_train,
                                                                                        running_avg_loss_train))
                start = time.time()

            if iter % 5000 == 0:
                running_avg_loss_eval = self.run_eval()
                write_summary("running_avg_loss_eval",
                              running_avg_loss_eval,
                              self.summary_writer_eval,
                              iter)
                self.summary_writer_eval.flush()
                self.chechpoint.check_loss(running_avg_loss_eval, running_avg_loss_train, iter)
                start = time.time()
                self.batcher_eval.start_threads()

            if config.is_coverage and iter % 2000 == 0:
                 self.chechpoint.save_model("coverage", running_avg_loss_eval, running_avg_loss_train, iter)
            if iter % 10000 == 0:
                self.chechpoint.save_model("critical", running_avg_loss_eval, running_avg_loss_train, iter)


if __name__ == '__main__':
    print(config.batch_size)
    train_processor = Train(vocab=None, vectors=None)
    train_processor.trainIters(config.max_iterations,
                               r"D:\PycharmProjects\pointer_summarizer\log_ria_gramm\new\train_1589144511\model\best\model_55000_1589168319")
