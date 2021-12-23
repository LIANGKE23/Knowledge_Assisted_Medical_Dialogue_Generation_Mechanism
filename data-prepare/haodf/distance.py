from typing import List
import numpy as np
from math import log10
import time
# import torch

class HanMingANDEdit(object):
    def __init__(self, faq):
        self.faq = faq
        self.word_bag = None
        self.faq2vec = None
        self.word_bag, self.faq2vec, self.word2faqidx = self._get_one_hot(self.faq)
        self.faq2vec = self.faq2vec

    @staticmethod
    def _get_one_hot(faq):
        # 构建词袋
        word_bag = []
        for q in faq:
            for w in q:
                if w not in word_bag:
                    word_bag.append(w)
        # one hot 向量
        faq2vec = []
        for kd in faq:
            v = [0]*len(word_bag)
            for w in kd:
                if w in word_bag:
                    v[word_bag.index(w)] = 1
            faq2vec.append(v)
        # 倒排索引
        word2faqidx = {}
        for i, q in enumerate(faq):
            for w in q:
                if w in word2faqidx:
                    word2faqidx[w].append(i)
                else:
                    word2faqidx[w] = [i]
        return word_bag, faq2vec, word2faqidx

    def hanming_distance(self, text: str = None) -> List:
#         t1=time.time()
        faq2vec_idx = []
        for w in text:
            if self.word2faqidx.get(w):
                for faqidx in self.word2faqidx[w]:
                    if faqidx not in faq2vec_idx:
                        faq2vec_idx.append(faqidx)
        faq2vec = [self.faq2vec[idx] for idx in faq2vec_idx]
        if not faq2vec:
            return []
        text2vec = [1.0 / len(text) if i in text else 0.0 for i in self.word_bag]
        # text2vec = torch.tensor(text2vec)
        # faq2vec = torch.FloatTensor(faq2vec)
        dis = np.dot(np.array(faq2vec), np.array(text2vec))
#         print("hm_dist: ", t1-time.time())
        # dis = torch.matmul(faq2vec, text2vec).numpy()
        most_sim = dis.argsort()
        res = []
        for i in reversed(most_sim):
            res.append((self.faq[faq2vec_idx[i]], dis[i]))
        return res

    def calculate(self, char2dict, text2vec, total):
        for w in char2dict:
            if w in self.word_bag:
                text2vec[self.word_bag.index(w)] = char2dict[w] / total

    @staticmethod
    def _min_edit_dist(sm, sn):
        m, n = len(sm) + 1, len(sn) + 1
        matrix = [[0] * n for i in range(m)]
        matrix[0][0] = 0
        for i in range(1, m):
            matrix[i][0] = matrix[i - 1][0] + 1
        for j in range(1, n):
            matrix[0][j] = matrix[0][j - 1] + 1
        for i in range(1, m):
            for j in range(1, n):
                if sm[i - 1] == sn[j - 1]:
                    cost = 0
                else:
                    cost = 1
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)
        return matrix[m - 1][n - 1]

    def edit_dist(self, text: str = None) -> List:
        min_dist = [self._min_edit_dist(i, text) for i in self.faq]
        most_sim = np.array(min_dist).argsort()
        res = []
        for i in most_sim:
            res.append((self.faq[i], min_dist[i]))
        return res

    def merge_dise(self, edit, hm): ## edit/10 - hm
        hm_edit_dis = {i[0]: i[1]/10 for i in edit}
        hm_edit_dis = {i[0]: hm_edit_dis[i[0]]-i[1] for i in hm}
        hm_edit_dis = sorted(hm_edit_dis.items(), key=lambda x: x[1])
        return hm_edit_dis

    def merge_symp(self, edit, hm): ## log(edit) - hm
        hm_edit_dis = {i[0]: i[1]/10 for i in edit}
        hm_edit_dis = {i[0]: hm_edit_dis[i[0]]-i[1] for i in hm}
        hm_edit_dis = sorted(hm_edit_dis.items(), key=lambda x: x[1])
        return hm_edit_dis


if __name__ == "__main__":
    import json
    with open("report_data.json", 'r', encoding='utf8') as f:
        report_data = json.load(f)
    items = []
    for key in report_data:
        items += key.split("/")

    dis = HanMingANDEdit(items)
    while True:
        value = input("输入")
        res1 = dis.hanming_distance(value)
        res2 = dis.edit_dist(value)
        res = dis.merge(res2, res1)
        print(res1)
        print(res2)
        print(res)









