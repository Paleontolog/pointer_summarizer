# from data_util.utils import *
from random import shuffle
#
#
# write_for_rouge(#[str("Привет земляне вашу мать".encode("UTF-8"))],
#                 ["Привет земляне вашу мать"],
#                 #list(map(lambda x: str(x.encode("UTF-8")),  ["мать", "вашу"])),
#                 ["мать", "вашу"],
#                 0,
#                 r"D:\PycharmProjects\pointer_summarizer\eval\ref", r"D:\PycharmProjects\pointer_summarizer\eval\sum")
#
# r = rouge_eval(r"D:\PycharmProjects\pointer_summarizer\eval\ref", r"D:\PycharmProjects\pointer_summarizer\eval\sum")
# print(r)
#


# from os import path
# import os
# #
# parent_dir = path.abspath(path.join(path.dirname(path.abspath(__file__)), os.pardir))
# t = pd.read_csv(f"{parent_dir}/data/lenta-ru-news-prepared.csv")
# # print("=--------------")
# t = t.sample(frac=1).reset_index(drop=True)
# print(t.head())
# import data_util.config
# from data_util.batcher import *
# from data_util.data import *
# import os
#

#if __name__ == "__main__":
    # file_list = glob.glob(r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_chunked_stemm\*")
    # shuffle(file_list)
    # test = file_list[:int(len(file_list) * 0.1)]
    # for f in test:
    #     new_name = f.replace("chunk_", "eval_chunk_")
    #     os.rename(f, new_name)

    # vocab = Vocab(config.vocab_path, config.vocab_size)
    # batcher_train = Batcher(config.train_data_path, vocab, mode='train',
    #                              batch_size=config.batch_size, single_pass=False)
    # ttt = 0
    # time.sleep(10)
    # batch = batcher_train.next_batch()
    # while batch:
    #     print(ttt)
    #     #print(batch.original_abstracts)
    #     ttt += 1
    #     batch = batcher_train.next_batch()

    # time.sleep(10)
    # batcher_train.start_threads()
    # time.sleep(10)


import pandas as pd
from os import path
import os
# #
# parent_dir = path.abspath(path.join(path.dirname(path.abspath(__file__)), os.pardir))
#
# with open(f"{parent_dir}/data/lenta-ru-news-prepared.txt", "r", encoding="utf-8") as r:
#     res = [tuple(line[1:-2].split("<p>")) for line in r.readlines()]
#     print(res[:10])
##t = pd.read_csv(f"{parent_dir}/data/lenta-ru-news-prepared.csv")

# t = t["title"] + "<p>" + t["text"]

# t.to_csv(f"{parent_dir}/data/lenta-ru-news-prepared.txt", index=False, sep=' ', header=None)


def chunk_data(file_path, save_path, chunk_len):
    with open(file_path, "r", encoding="utf-8") as r:
        data = [line[1:-2] for line in r.readlines()]
    shuffle(data)
    for idx in range(len(data) // chunk_len + 1):
        with open(f"{save_path}/chunk_{idx}", "w", encoding="utf-8") as w:
            w.write("\n".join(data[idx * chunk_len: (idx + 1) * chunk_len]))

def get_BPE_vocab(model_path=r"D:\PycharmProjects\pointer_summarizer\tokenize_ria.model",
                 vocab_path=r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\ria_dict_gramm.txt"):
    import youtokentome as yttm
    bpe = yttm.BPE(model=model_path)
    vocab = bpe.vocab()[4:]
    result = []
    for w in vocab:
        result.append(w + " " + str(1))

    with open(vocab_path, 'w', encoding="utf-8") as w:
        w.write("\n".join(result))

if __name__ == "__main__":
    parent_dir = path.abspath(path.join(path.dirname(path.abspath(__file__)), os.pardir))
    chunk_data(f"{parent_dir}/data/lenta-ru-news-prepared.txt",
               f"{parent_dir}/data/chunked_lenta_news", 1000)

    with open(f"{parent_dir}/data/chunked_lenta_news/chunk_2", "r", encoding="utf-8") as r:
        print("\n".join([str(line.split("<p>")) for line in r.readlines()]))
