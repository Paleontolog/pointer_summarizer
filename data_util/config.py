import os

root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
#train_data_path = os.path.join(root_dir, "C:/cnn-dailymail-master/finished_files/chunked/train_*")
#train_data_path = os.path.join(root_dir, "C:/cnn-dailymail-master/finished_files/chunked/train_*")
#train_data_path = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\data\chunked_lenta_news\chunk_*")
#train_data_path = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_chunked_lemmatize\chunk_*")
#eval_data_path = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_chunked_lemmatize\eval_chunk_*")


decode_data_path = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_chunked_lemmatize\eval_chunk_13")

# decode_data_path = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\data\LENTA\chunked_lenta_news\chunk_13")


#vocab_path = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\data\LENTA\base_model_result_vocab.txt")
#vocab_path = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\ria_dict_lemmatize_result_fasttext.txt")
vocab_path = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\ria_dict_stemm_result.txt")
#vocab_path = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\ria_dict_gramm.txt")
#vocab_path = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\ria_dict_stemm_result.txt")

#log_root = os.path.join(root_dir, r"D:\PycharmProjects\pointer_summarizer\w2v_model\base")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 16
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=20
vocab_size= 100000#69996

#lr=0.15
lr=0.001
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = True
cov_loss_wt = 1.0


eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.001#0.15


print_interval = 100
eval_interval = 1000
save_interval = 5000