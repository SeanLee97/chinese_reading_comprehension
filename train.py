# /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import time
import libs
import numpy as np
from libs.Dataset import Dataset
from aoa import AOA
from config.config import *
from libs.Model import *

if USE_CUDA:
    torch.cuda.set_device(N_GPU)

def loss_func(answers, pred_answers, answer_probs):
    num_correct = (answers == pred_answers).sum().squeeze().data[0]
    loss = - torch.mean(torch.log(answer_probs))
    if USE_CUDA:
        return loss.cuda(), num_correct
    else:
        return loss, num_correct

def eval(model, data):
    total_loss = 0
    total = 0
    total_correct = 0

    model.eval()
    for i in range(len(data)):
        (batch_docs, batch_docs_len, doc_mask), (batch_querys, batch_querys_len, query_mask), batch_answers, candidates = data[i]

        pred_answers, probs = model(batch_docs, batch_docs_len, doc_mask,
                                    batch_querys, batch_querys_len, query_mask,
                                    answers=batch_answers, candidates=candidates)

        loss, num_correct = loss_func(batch_answers, pred_answers, probs)

        total_in_minibatch = batch_answers.size(0)
        total_loss += loss.data[0] * total_in_minibatch
        total_correct += num_correct
        total += total_in_minibatch

        del loss, pred_answers, probs

    model.train()
    return total_loss / total, total_correct / total

def train_model(model, trainData, validData, optimizer: torch.optim.Adam):
    start_time = time.time()
    model, optimizer, start_epoch = load_previous_model(model, optimizer)

    def train_epoch(epoch):
        trainData.shuffle()
        total_loss, total, total_num_correct = 0, 0, 0
        report_loss, report_total, report_num_correct = 0, 0, 0
        for i in range(len(trainData)):
            (batch_docs, batch_docs_len, doc_mask), (batch_querys, batch_querys_len, query_mask), batch_answers, candidates = trainData[i]

            model.zero_grad()
            pred_answers, answer_probs = model(batch_docs, batch_docs_len, doc_mask, batch_querys, batch_querys_len, query_mask,answers=batch_answers, candidates=candidates)

            loss, num_correct = loss_func(batch_answers, pred_answers, answer_probs)

            loss.backward()
            for parameter in model.parameters():
                parameter.grad.data.clamp_(-5.0, 5.0)

            optimizer.step()

            total_in_minibatch = batch_answers.size(0)

            report_loss += loss.data[0] * total_in_minibatch
            report_num_correct += num_correct
            report_total += total_in_minibatch

            total_loss += loss.data[0] * total_in_minibatch
            total_num_correct += num_correct
            total += total_in_minibatch
            if i % LOG_INTERVAL == 0:
                print("Epoch %2d, %5d/%5d; avg loss: %.2f; acc: %6.2f%%;  %6.0f s elapsed" %
                      (epoch, i+1, len(trainData),
                       report_loss / report_total,
                       report_num_correct / report_total * 100,
                       time.time()-start_time))

                report_loss = report_total = report_num_correct = 0
            del loss, pred_answers, answer_probs

        return total_loss / total, total_num_correct / total

    for epoch in range(start_epoch, EPOCHS + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_acc = train_epoch(epoch)
        print('Epoch %d:\t average loss: %.2f\t train accuracy: %g' % (epoch, train_loss, train_acc*100))

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, validData)
        print('=' * 20)
        print('evaluating on validation set:')
        print('Validation loss: %.2f' % valid_loss)
        print('Validation accuracy: %g' % (valid_acc*100))
        print('=' * 20)

        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch,
            'optimizer': optimizer_state_dict,
        }
        torch.save(checkpoint, '%s/rd_epoch-%d.pt' % (MODEL_DIR, epoch))
   
def main():
    if USE_CUDA:
        print("GPU accelerating ...")


    vocab_dict = torch.load(DICT_PATH)
    train_data = torch.load(TRAIN_DATA_PATH)
    valid_data = torch.load(VALID_DATA_PATH)

    train_dataset = Dataset(train_data, BATCH_SIZE, N_GPU)
    valid_dataset = Dataset(valid_data, BATCH_SIZE, N_GPU, volatile=True)
    
    print(' * vocabulary size = %d' %
          (vocab_dict.size()))
    print(' * number of training samples. %d' %
          len(train_data['answers']))
    print(' * maximum batch size. %d' % BATCH_SIZE)

    print('Building model...')

    model = AOA(vocab_dict, dropout_rate=DROPOUT, embed_dim=EMBED_SIZE, hidden_dim=HIDDEN_SIZE)

    if USE_CUDA:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)


    train_model(model, train_dataset, valid_dataset, optimizer)

if __name__ == '__main__':
    main()
