import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F
import torch.nn.init as weigth_init
from config.config import *


def sort_batch(data, seq_len):
    sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
    sorted_data = data[sorted_idx.data]
    _, reverse_idx = torch.sort(sorted_idx, dim=0, descending=False)
    if USE_CUDA:
        return sorted_data, sorted_seq_len.cuda(), reverse_idx.cuda()
    else:
        return sorted_data, sorted_seq_len, reverse_idx




def softmax_mask(input, mask, axis=1, epsilon=1e-12):
    shift, _ = torch.max(input, axis)
    shift = shift.expand_as(input)
    if USE_CUDA:
        shift.cuda()

    target_exp = torch.exp(input - shift) * mask

    normalize = torch.sum(target_exp, axis).expand_as(target_exp)
    softm = target_exp / (normalize + epsilon)
    if USE_CUDA:
        return softm.cuda()
    else:
        return softm


class AOA(nn.Module):

    def __init__(self, vocab_dict, dropout_rate, embed_dim, hidden_dim, bidirectional=True):
        super(AOA, self).__init__()
        self.vocab_dict = vocab_dict
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(vocab_dict.size(),
                                      self.embed_dim,
                                      padding_idx=PAD)
        self.embedding.weight.data.uniform_(-0.05, 0.05)

        input_size = self.embed_dim
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_dim, dropout=dropout_rate,
                          bidirectional=bidirectional, batch_first=True)

        for weight in self.gru.parameters():
            if len(weight.size()) > 1:
                weigth_init.orthogonal(weight.data)

    def forward(self, docs_input, docs_len, doc_mask,
                      querys_input, querys_len, query_mask,
                      candidates=None, answers=None):
        s_docs, s_docs_len, reverse_docs_idx = sort_batch(docs_input, docs_len)
        s_querys, s_querys_len, reverse_querys_idx = sort_batch(querys_input, querys_len)

        docs_embedding = pack(self.embedding(s_docs), list(s_docs_len.data), batch_first=True)
        querys_embedding = pack(self.embedding(s_querys), list(s_querys_len.data), batch_first=True)

        # encode
        docs_outputs, _ = self.gru(docs_embedding, None)
        querys_outputs, _ = self.gru(querys_embedding, None)

        # unpack
        docs_outputs, _ = unpack(docs_outputs, batch_first=True)
        querys_outputs, _ = unpack(querys_outputs, batch_first=True)

        docs_outputs = docs_outputs[reverse_docs_idx.data]
        querys_outputs = querys_outputs[reverse_querys_idx.data]


        # transpose query for pair-wise dot product
        dos = docs_outputs
        doc_mask = doc_mask.unsqueeze(2)
        qos = torch.transpose(querys_outputs, 1, 2)
        query_mask = query_mask.unsqueeze(2)

        # pair-wise matching score
        M = torch.bmm(dos, qos)
        M_mask = torch.bmm(doc_mask, query_mask.transpose(1, 2))

        # query-document attention
        alpha = softmax_mask(M, M_mask, axis=1)
        beta = softmax_mask(M, M_mask, axis=2)

        sum_beta = torch.sum(beta, dim=1)

        docs_len = docs_len.unsqueeze(1).unsqueeze(2).expand_as(sum_beta)
        average_beta = sum_beta / docs_len.float()


        # attended document-level attention
        s = torch.bmm(alpha, average_beta.transpose(1, 2))

        # predict the most possible answer from given candidates
        pred_answers = None
        #pred_locs = None
        probs = None
        if candidates is not None:
            pred_answers = []
            pred_locs = []
            for i, cands in enumerate(candidates):
                pb = []
                # 维度压缩至一维
                document = docs_input[i].squeeze()
                for j, candidate in enumerate(cands):
                    pointer = (document == candidate.expand_as(document))
                    pb.append(torch.sum(s[i][pointer]))
                pb = torch.cat(pb, dim=0).squeeze()
                #print("pb> ", pb)
                _ , max_loc = torch.max(pb, 0)
                pred_answers.append(cands.index_select(0, max_loc))
                #pred_locs.append(max_loc)
            pred_answers = torch.cat(pred_answers, dim=0).squeeze()
            '''
            print("old pred_answers>", self.vocab_dict.convert2word(pred_answers.data))
            print('pred_answers > ', pred_answers)
            pred_locs = torch.cat(pred_locs, dim=0).squeeze()
            print('pred_locs > ', pred_locs)
            '''

        if answers is not None:
            probs = []
            for i, answer in enumerate(answers):
                document = docs_input[i].squeeze()
                pointer = document == answer.expand_as(document)
                probs.append(torch.sum(s[i][pointer]))
            probs = torch.cat(probs, 0).squeeze()

        return pred_answers, probs







