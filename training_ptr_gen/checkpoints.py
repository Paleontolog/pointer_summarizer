import os
import torch
import time
import glob

class Checkpoint:
    def __init__(self, model, optimizer, model_dir,
                 best_loss, patience=4,
                 top_n=5, delta=0.0):
        self.model = model
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.best_loss = best_loss
        self.counter = 0
        self.patience = patience
        self.top_n = top_n
        self.delta = delta
        print(f"CUR BEST {self.best_loss}")

    def save_model(self, type, running_avg_loss_eval, running_avg_loss_train, iter):
        if not os.path.exists(f"{self.model_dir}/{type}"):
            os.mkdir(f"{self.model_dir}/{type}")

        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_train_loss': running_avg_loss_train,
            'current_eval_loss': running_avg_loss_eval
        }

        model_save_path = os.path.join(f"{self.model_dir}/{type}", f"model_{iter}_{int(time.time())}")
        torch.save(state, model_save_path)

    def check_loss(self, running_avg_loss_eval, running_avg_loss_train, iter):
        print(f"BEST LOSS: {self.best_loss}")
        print(f"CURR LOSS: {running_avg_loss_eval}")
        stop_train = False
        if running_avg_loss_eval < self.best_loss + self.delta:
            self.best_loss = running_avg_loss_eval
            self.counter = 0
            prev_model = glob.glob(f"{self.model_dir}/best/*")
            if prev_model:
                print(f"REMOVE MODEL {prev_model[0]}")
                os.remove(prev_model[0])
            self.save_model("best", running_avg_loss_eval, running_avg_loss_train, iter)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                stop_train = True

        if stop_train:
            self.save_model("stop", running_avg_loss_eval, running_avg_loss_train, iter)
        else:
            self.save_model("last", running_avg_loss_eval, running_avg_loss_train, iter)
            model_list = glob.glob(f"{self.model_dir}/last/*")
            if len(model_list) >= self.top_n:
                model_list = sorted(model_list, key=lambda x: int(x.split("\\")[-1].split("_")[1]))
                os.remove(model_list[0])
                print(f"REMOVE LAST {model_list[0]}")

        return stop_train

