# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
配置文件
'''
import torch

USE_CUDA 			= torch.cuda.is_available()
N_GPU = 0
CORPUS_DIR	 		= './corpus'
RUNTIME_DIR 		= './runtime'

PAD = 0
UNK = 1
PAD_WORD = "<PAD>"
UNK_WORD = "<UNK>"

HIDDEN_SIZE = 512
EMBED_SIZE = 512
DROPOUT = 0.1
BATCH_SIZE = 10
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0

LOG_INTERVAL = 10

DICT_PATH = RUNTIME_DIR + '/dict.pt'
MODEL_DIR = RUNTIME_DIR + '/model'
TRAIN_DATA_PATH = RUNTIME_DIR + '/train.pt'
TEST_DATA_PATH = RUNTIME_DIR + '/test.pt'
VALID_DATA_PATH = RUNTIME_DIR + '/valid.pt'

