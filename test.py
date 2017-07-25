from libs.Dataset import Dataset
import torch
import time
import os
from aoa import AOA
from preprocess import get_stories, vectorize_stories
from config.config import *
from libs.Model import *

def evalulate(model, data, vocab_dict):

    def acc(answers, pred_answers):
        num_correct = (answers == pred_answers).sum().squeeze().data[0]
        return num_correct

    model.eval()
    answers = []
    total_correct = 0
    total = 0
    for i in range(len(data)):
        (batch_docs, batch_docs_len, doc_mask), (batch_querys, batch_querys_len, query_mask), batch_answers , candidates = data[i]

        pred_answers, _ = model(batch_docs, batch_docs_len, doc_mask,
                                    batch_querys, batch_querys_len, query_mask,
                                    candidates=candidates, answers=batch_answers)

        answers.extend(pred_answers.data)
        num_correct = acc(batch_answers, pred_answers)
        total_in_minibatch = batch_answers.size(0)
        total_correct += num_correct
        total += total_in_minibatch
        del pred_answers

    print("Evaluating on test set:\nAccurary {:.2%}".format(total_correct / total))
    print(vocab_dict.convert2word(answers))
    return vocab_dict.convert2word(answers)

def main():
    vocab_dict = torch.load(DICT_PATH)

    print("Loading test data")
    test_data = torch.load(TEST_DATA_PATH)
    
    model = AOA(vocab_dict, dropout_rate=DROPOUT, embed_dim=EMBED_SIZE, hidden_dim=HIDDEN_SIZE)
    model, _, start_epoch = load_previous_model(model)
    if USE_CUDA:
        model.cuda()

    test_dataset = Dataset(test_data, BATCH_SIZE, N_GPU, volatile=True)

    print('Evaluate on test data')
    answers = evalulate(model, test_dataset, vocab_dict)

if __name__ == '__main__':
    main()


