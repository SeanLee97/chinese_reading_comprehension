# /usr/bin/env python
# -*- coding: utf-8 -*-
import json,time,os,re
import numpy as np
import torch
import itertools
import jieba
import jieba.posseg
import jieba.analyse
from functools import reduce
from joblib import Parallel, delayed
from libs.Dict import Dict as Vocabulary
from config.config import *
from aoa import AOA
from preprocess import vectorize_stories
from libs.Dataset import Dataset
from libs.Similarity import Similarity
from config.config import *
from libs.Model import *


data_path = CORPUS_DIR
vocab_file = os.path.join(RUNTIME_DIR, 'vocab.json')
dict_file = os.path.join(RUNTIME_DIR, 'dict.pt')

vocab_dict = torch.load(dict_file)
model = AOA(vocab_dict, dropout_rate=DROPOUT, embed_dim=EMBED_SIZE, hidden_dim=HIDDEN_SIZE)
model, _, start_epoch = load_previous_model(model)
if USE_CUDA:
    model.cuda()
        
def vectorize_stories(stories, vocab : Vocabulary):
    X = []
    Q = []
    C = []
    A = []
    for s, q, a, c in stories:
        '''
        print("story> ", s)
        print("question> ", q)
        print("answer> ", a)
        print("candidates> ", c)
        '''
        x = vocab.convert2idx(s)
        xq = vocab.convert2idx(q)
        xc = vocab.convert2idx(c)
        X.append(x)
        Q.append(xq)
        C.append(xc)
        A.append(vocab.getIdx(a))
    
    X = X
    Q = Q
    C = C
    A = torch.LongTensor(A)
    return X, Q, A, C

def evalulate(model, data, vocab_dict):
    model.eval()
    answers = []
    for i in range(len(data)):
        (batch_docs, batch_docs_len, doc_mask), (batch_querys, batch_querys_len, query_mask), batch_answers , candidates = data[i]

        pred_answers, _ = model(batch_docs, batch_docs_len, doc_mask,
                                    batch_querys, batch_querys_len, query_mask,
                                    candidates=candidates, answers=batch_answers)

        answers.extend(pred_answers.data)
        print(pred_answers)
        del pred_answers
    return vocab_dict.convert2word(answers)

def rm_sign(strs):
    return re.sub("[\s+\.\!\/<>“”,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", strs.strip())

def word_tokenize(sentence):
    return jieba.lcut(sentence)

def tokenize(sentence):
    sentence = sentence.replace("|", "")
    sentence = rm_sign(sentence)
    return [s.strip() for s in word_tokenize(sentence) if s.strip ()]

def get_n(sentence):
    words = jieba.posseg.cut(sentence)
    word_list = []
    for word, flag in words:
        if 'n' in flag or flag in ['vn']:
            word_list.append(word)
    return set(word_list)

def get_sentences(sentence):
    sent_list = sentence.strip().split("|")
    result_list = []
    for sent in sent_list:
        if len(sent) > 0:
            result_list.append(word_tokenize(sent.strip()))
    return result_list

def get_candidate(question, sentence):
    sentences = sentence.strip().split("|")
    candidate = {'sim':0, 'val': ''}
    for k, v in enumerate(sentences):
        if len(v) > 0:
            sim = Similarity(question, v).cosSim()
            if sim >= candidate['sim']:
                candidate['sim'] = sim
                candidate['val'] = v
    question_list = tokenize(question)
    candi_tmp = get_n(candidate['val'])
    candidate_list = [x for x in candi_tmp if x not in question_list]
    return candidate_list

def predict(story_raw, question_raw):
    story_list = tokenize(story_raw)
    question_list = tokenize(question_raw)
    candidate_list = get_candidate(question_raw, story_raw)
    #print(candidate_list)
    story = [(story_list, question_list, '', list(candidate_list))]
    test = {}
    test['documents'], test['querys'], test['answers'], test['candidates'] = vectorize_stories(story, vocab_dict)
    test_dataset = Dataset(test, BATCH_SIZE, N_GPU, volatile=True)
    answers = evalulate(model, test_dataset, vocab_dict)
    return answers[0]

def main():
    story = input("input trained story> ")
    while True:
        question = input("input your question> ")
        answer = predict(story, question)
        print(answer)
    
if __name__ == '__main__':
    main()
