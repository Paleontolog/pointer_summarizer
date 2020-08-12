from sumeval.metrics.rouge import RougeCalculator
import re
from nltk.corpus import stopwords
from sumeval.metrics.lang.base_lang import BaseLang
import pymystem3
from nltk.translate import bleu

class LangRU(BaseLang):

    def __init__(self):
        super(LangRU, self).__init__("ru")
        self._symbol_replace = re.compile(r"[^А-Яа-я0-9-]")
        self._valid_word = re.compile(r"^[А-Яа-я0-9$]")

    def tokenize(self, text):
        return text.split(" ")

    def tokenize_with_preprocess(self, text):
        _text = self._preprocess(text)
        words = self.tokenize(_text)
        words = [w.strip() for w in words if w.strip()]
        words = [w for w in words if self._valid_word.match(w)]
        return words

    def _preprocess(self, text):
        _text = text.replace("-", " - ")
        _text = self._symbol_replace.sub(" ", _text)
        _text = _text.strip()
        return _text

    def load_stopwords(self):
        self._stopwords = set(stopwords.words('russian'))


def print_results(result_rouge):
    print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-3: {}, ROUGE-L: {},  BLEU-2: {}, BLEU-3: {}".format(
        result_rouge[0], result_rouge[1], result_rouge[2],
        result_rouge[3], result_rouge[4], result_rouge[5]
    ).replace(", ", "\n"))



def calk_rouge(summary, references, rouge, lemmatizer):
    summary = "".join(lemmatizer.lemmatize(summary)).strip()
    references = ["".join(lemmatizer.lemmatize(ref)).strip() for ref in references]

    rouge_n = []
    for n in range(1, 4):
        rouge_n.append(rouge.rouge_n(summary=summary, references=references, n=n))

    rouge_l = rouge.rouge_l(
        summary=summary,
        references=references)

    bleu_score_2 = bleu(references, summary,  weights=(1./2., 1./2.))
    bleu_score_3 = bleu(references, summary, weights=(1./3., 1./3., 1./3.))

    print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-3: {}, ROUGE-L: {}, BLEU-2: {}, BLEU-3: {}".format(
        rouge_n[0], rouge_n[1], rouge_n[2], rouge_l, bleu_score_2, bleu_score_3
    ).replace(", ", "\n"))

    return rouge_n + [rouge_l, bleu_score_2, bleu_score_3]


if __name__ == "__main__":
    lemm = pymystem3.Mystem()
    rouge = RougeCalculator(stopwords=True, lang=LangRU())
    calk_rouge("Михаил Горбачев решил побороться за реформы в Индонезии",
               ["В Индонезии решил баллотироваться политик по имени хороший Гитлер и заговорил о "],
               rouge,
               lemm)