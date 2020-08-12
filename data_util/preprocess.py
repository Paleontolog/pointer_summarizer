from nltk.stem.snowball import RussianStemmer
import glob
from multiprocessing import Pool
import re
import youtokentome
from collections import Counter
import pymystem3.mystem as mystem
import os
from nltk.tokenize import sent_tokenize, word_tokenize


def split_to_stem_and_gram(word, stemmer):
    stem = stemmer.stem(word)
    flex = word[len(stem):]
    return f"{stem} +{flex}" if flex else stem

def split_all_words(sent, stemmer):
    return " ".join([split_to_stem_and_gram(word, stemmer) for word in sent.split()])


def prepare_stem_data(line, stemmer):
    return "<p>".join([split_all_words(item, stemmer) for item in line.split("<p>") ])


def process_file_stemm(file_name):
    print(file_name)
    cnt = 0
    result = []
    stemmer = RussianStemmer(False)
    with open(file_name, "r", encoding="utf-8") as r:
        for line in r.readlines():
            try:
                line = prepare_stem_data(line, stemmer)
                result.append(line)
            except:
                cnt += 1
    print(f"Bad lines: {cnt}")
    with open(file_name.replace("chunked_lenta_news", "chunked_lenta_news_gramms"), "w", encoding="utf-8") as w:
        w.write("\n".join(result))


def process_file_tokens(file_name,
                        model_path=r"D:\PycharmProjects\pointer_summarizer\tokenize_ria.model"):
    print(file_name)
    result = []
    cnt = 0
    token_model = youtokentome.BPE(model=model_path)
    with open(file_name, "r", encoding="utf-8") as r:
        for line in r.readlines():
            try:
                line = line.replace("<s>", " ").replace("</s>", " ").strip()
                line = token_model.encode(line.split("<p>"),
                                          output_type=youtokentome.OutputType.SUBWORD)
                line = [" ".join(subline) for subline in line]
                line[0] = "<s> " + line[0] + " </s>"
                line = "<p>".join(line)
                result.append(line)
            except:
                cnt += 1
    print(f"Bad lines: {cnt}")
    with open(file_name.replace("ria_chunked", "ria_chunked_gramm"), "w", encoding="utf-8") as w:
        w.write("\n".join(result))


def create_result_dict(from_dict=r"D:\PycharmProjects\pointer_summarizer\data\chunked_lenta_dict.txt",
                       to_dict=r"D:\PycharmProjects\pointer_summarizer\data\chunked_lenta_dict_prepare.txt"):
    pattern = re.compile(r".*[a-zA-Z0-9]+.*")
    with open(from_dict, "r", encoding="utf-8") as r:
        vocab = dict()
        for i, w in enumerate(r.readlines()):
            w = w.strip().split(" ")
            if pattern.match(w[0]) is not None:
                print(w)
            elif w[0] not in vocab:
                vocab[w[0].strip()] = int(w[1].strip())
            if len(vocab) == 150000:
                break
        with open(to_dict, "w", encoding="utf-8") as file:
            for w, i in vocab.items():
                file.write(f"{str(w).strip()} {i}")
                file.write("\n")


def create_vocab_from_file(file_name):
    print(file_name)
    dict = Counter()
    with open(file_name, "r", encoding="utf-8") as r:
        for line in r.readlines():
            dict.update(line.replace("<p>", " ").split(" "))
    return dict



def create_vocab(pool, file_list,
                 save_path=r"D:\PycharmProjects\pointer_summarizer\data\chunked_lenta_dict.txt"):
    result_dict = Counter()
    result = pool.map(create_vocab_from_file, file_list)

    for d in result:
        result_dict.update(d)

    with open(save_path, "w", encoding="utf-8") as file:
        for w, i in result_dict.most_common():
            file.write(f"{str(w).strip()} {i}")
            file.write("\n")


def lemmatize_chunks(file_name):
    print(file_name)
    lemmatizer = mystem.Mystem()
    with open(file_name, "r", encoding="utf-8") as r:
        file = r.read()
        file = file.replace("<s>", " <s> ")\
            .replace("</s>", " </s> ")\
            .replace("<p>", " <p> ")\
            .replace("\n", "<n>")
        file = "".join(lemmatizer.lemmatize(file))
        file = file.replace("<n>", "\n")
    with open(file_name.replace("ria_chunked", "ria_chunked_lemmatize"), "w", encoding="utf-8") as w:
        w.write(file)


def create_test(all_files, names_file, os_delimer="/"):
    with open(names_file, "r") as r:
        eval_list = set([l.strip() for l in r.readlines()])
        print(all_files)

        new_eval = list(filter(lambda x: x.split(os_delimer)[-1] in eval_list, all_files))
        for f in new_eval:
            new_name = f.replace("chunk_", "eval_chunk_")
            os.rename(f, new_name)


def preprocess_stemm(text, stemmer):
    text = text.split()
    text = [split_to_stem_and_gram(word, stemmer) for word in text]
    return " ".join(text)


def preprocess_lemm(text, lemmatizer):
    return "".join(lemmatizer.lemmatize(text))


def preprocess_gramm(text, token_model):
    text = token_model.encode(text, output_type=youtokentome.OutputType.SUBWORD)
    return " ".join(text)



if __name__ == '__main__':
    file_list = glob.glob(r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_chunked\*")
    # create_test(file_list,
    #             r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_chunked_stemm_train.txt",
    #             os_delimer="\\")
    pool = Pool(processes=10)

    create_vocab(pool, file_list,
                 r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_raw_dict.txt")
    #pool = Pool(processes=10)
    #file_list = glob.glob(r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_chunked_lemmatize\*")
    #result = pool.map(lemmatize_chunks, file_list)
    #create_vocab(file_list, r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_dict_lemmatize.txt")
    # create_result_dict(r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\ria_dict_lemmatize.txt",
    #                    r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\ria_dict_lemmatize_result.txt")

    # l = glob.glob(r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_chunked_stemm\*")
    # res = list(filter(lambda x: "eval" in x, l))
    # res = list(map(lambda x: x.split("\\")[-1], res))
    # res = list(map(lambda x: x.replace("eval_", ""), res))
    # with open(r"D:\PycharmProjects\pointer_summarizer\data\chunked_ria\content\ria_chunked_stemm_train.txt", "w") as w:
    #     w.write("\n".join(res))
    # print(res)
