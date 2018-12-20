# -*- coding: utf-8 -*-

import os
import math
import gensim
import time
import sys
import json
import copy
import operator

import numpy as np
from scipy.misc import logsumexp
from random import shuffle

from scipy.sparse import hstack, vstack
from collections import defaultdict
from collections import Counter
from gensim.models import word2vec
from numpy import unravel_index
from sklearn import metrics


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

    
# MODEL
class PCNN_ATT(nn.Module):
    
    def __init__(self, pretrained_word_emb, opt):
        
        super(PCNN_ATT, self).__init__()
        
        self.emb_dim = opt['emb_dim']
        self.pos_dim = opt['pos_dim']
        
        d_all = self.emb_dim + 2 * self.pos_dim
        
        self.win_size = opt['win_size']
        self.num_conv = opt['num_conv']

        self.pos_e1_size = opt['pos_e1_size']
        self.pos_e2_size = opt['pos_e2_size']
        self.pos_min_e1 = opt['pos_min_e1']
        self.pos_min_e2 = opt['pos_min_e2']

        self.dropout = opt['dropout']
        
        self.word_embedding = nn.Embedding(opt['vocab_size'], self.emb_dim)
        self.pos_embedding_e1 = nn.Embedding(opt['pos_e1_size'], self.pos_dim)
        self.pos_embedding_e2 = nn.Embedding(opt['pos_e2_size'], self.pos_dim)
        
        
        self.conv1 = nn.Conv2d(1, self.num_conv, (self.win_size, d_all), padding=(self.win_size-1, 0))
        self.linear = nn.Linear(self.num_conv*3, opt['num_rel'])
        
        self.param_initialization(pretrained_word_emb)
        
        
    def norm2(self, mat):
        v = torch.from_numpy(mat)
        v = F.normalize(v, p=2, dim=1)
        return v

        
    def param_initialization(self, pretrained_word_emb):
        
        # Word embedding is pretrained, and others are randomly initialized
        pretrained_word_emb = np.concatenate((np.random.uniform(-1, 1, size=(1, self.emb_dim)), pretrained_word_emb), axis=0)
        pos_emb1 = np.random.uniform(-1, 1, size=(self.pos_e1_size, self.pos_dim))
        pos_emb2 = np.random.uniform(-1, 1, size=(self.pos_e2_size, self.pos_dim))
        
        # Do p_2norm (following c++ code)
        pretrained_word_emb = self.norm2(pretrained_word_emb)
        pos_emb1 = self.norm2(pos_emb1)
        pos_emb2 = self.norm2(pos_emb2)
        
        # Copy data matrix to parameters
        self.word_embedding.weight.data.copy_(pretrained_word_emb)
        self.pos_embedding_e1.weight.data.copy_(pos_emb1)
        self.pos_embedding_e2.weight.data.copy_(pos_emb2)
        
        # Only matrix uses xavier_uniform, bias not
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.uniform_(self.linear.bias)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.uniform_(self.conv1.bias)
        
    
          
    def pairwise_pooling(self, sen_em, e1, e2):
        
        # As we pad (win_size-1) at the start and end of the sentence,
        # there is a "+ self.win_size" when calculate the index
        e1_posi = np.where(e1 == -self.pos_min_e1)[0][0] + self.win_size 
        e2_posi = np.where(e2 == -self.pos_min_e2)[0][0] + self.win_size
        
        if e1_posi > e2_posi:
            e1_posi, e2_posi = e2_posi, e1_posi
        
        sen_em1 = sen_em[:, :, :e1_posi, :]
        sen_em2 = sen_em[:, :, e1_posi:e2_posi, :]
        
        try:
            sen_em3 = sen_em[:, :, e2_posi:, :]
        except:
            print e1_posi, e2_posi, sen_em.size()
            raise
        
        # pw_result is a (num_conv * 3) matrix
        pw_result = torch.cat([F.max_pool2d(sen_em1, kernel_size=sen_em1.size()[2:]).view(-1, 1), 
                                   F.max_pool2d(sen_em2, kernel_size=sen_em2.size()[2:]).view(-1, 1),
                                   F.max_pool2d(sen_em3, kernel_size=sen_em3.size()[2:]).view(-1, 1)], 1)

        return pw_result
        
     
    
    
    def forward(self, sen_list, target, all_data):
        
        for index, sentence in enumerate(sen_list):

            sig_sentence = autograd.Variable( torch.LongTensor(all_data.train_list[sentence]).cuda() )
            pos_e1 = autograd.Variable( torch.LongTensor(all_data.train_pos_e1[sentence]).cuda() )
            pos_e2 = autograd.Variable( torch.LongTensor(all_data.train_pos_e2[sentence]).cuda() )
            
            words_embeds = self.word_embedding(sig_sentence)
            pos_e1_embeds = self.pos_embedding_e1(pos_e1)
            pos_e2_embeds = self.pos_embedding_e2(pos_e2)

            sentence_embeds = torch.cat([words_embeds, pos_e1_embeds, pos_e2_embeds], 1).unsqueeze(0).unsqueeze(0)
            
            sentence_embeds = self.conv1(sentence_embeds)
            
            # The result sentence_embeds is a (num_conv * 3) matrix
            sentence_embeds = self.pairwise_pooling(sentence_embeds, pos_e1.data.cpu().numpy(), pos_e2.data.cpu().numpy())
                
            sentence_embeds = torch.tanh(sentence_embeds).view(1, -1)

            if index == 0:
                bag_embeds = sentence_embeds
            else:
                bag_embeds = torch.cat([bag_embeds, sentence_embeds], 0)
                
        # att_matrix is a (sen_num * relation_num) matrix
        # att_matrix = F.softmax(0.5 * self.linear(bag_embeds), dim = 0)
        att_matrix = F.softmax(self.linear(bag_embeds), dim = 0)
        
        # bag_matrix_rel is a (relation_num * fea_num) matrix
        bag_matrix_rel = torch.t(att_matrix).mm(bag_embeds)

        # Select the target relation representation
        final_embeds = F.dropout(bag_matrix_rel[target, :].unsqueeze(0), p=self.dropout, training=True)
        
        # Since dropout layer adds a scalar 1/(1-p), 0.5 is to keep
        # the same with c++ code
        # final_embeds = self.linear(0.5 * final_embeds)
        final_embeds = self.linear(final_embeds)

        
        return F.log_softmax(final_embeds, dim = 1)
    
    

    def test(self, all_data):
        
        # start_time = time.time()

        prob_label_result = []
        

        for index, bag_name in enumerate(all_data.bags_test.keys()[:]):

            # if index > 0 and index % 20000 == 0:
            #     print 'index == ', index 

            sen_list = all_data.bags_test[bag_name]
            
            target = list(set([all_data.test_rel[sen] for sen in sen_list]))
          
            
            for index, sentence in enumerate(sen_list):
                
                sig_sentence = autograd.Variable( torch.LongTensor(all_data.test_list[sentence]).cuda() )
                pos_e1 = autograd.Variable( torch.LongTensor(all_data.test_pos_e1[sentence]).cuda() )
                pos_e2 = autograd.Variable( torch.LongTensor(all_data.test_pos_e2[sentence]).cuda() )
            
                words_embeds = self.word_embedding(sig_sentence)
                pos_e1_embeds = self.pos_embedding_e1(pos_e1)
                pos_e2_embeds = self.pos_embedding_e2(pos_e2)
                
                sentence_embeds = torch.cat([words_embeds, pos_e1_embeds, pos_e2_embeds], 1).unsqueeze(0).unsqueeze(0)
                
                sentence_embeds = self.conv1(sentence_embeds)
                
                sentence_embeds = self.pairwise_pooling(sentence_embeds, pos_e1.data.cpu().numpy(), pos_e2.data.cpu().numpy())
                
                sentence_embeds = torch.tanh(sentence_embeds).view(1, -1)

                if index == 0:
                    bag_embeds = sentence_embeds
                else:
                    bag_embeds = torch.cat([bag_embeds, sentence_embeds], 0)
                    
            # att_matrix = F.softmax(0.5 * self.linear(bag_embeds), dim = 0)
            att_matrix = F.softmax(self.linear(bag_embeds), dim = 0)
            
            bag_matrix_rel = torch.t(att_matrix).mm(bag_embeds)
            
            # prob_matrix = F.softmax(0.5 * self.linear(bag_matrix_rel), dim = 1)
            prob_matrix = F.softmax(self.linear(bag_matrix_rel), dim = 1)
    
            final_prob = prob_matrix.data.cpu().numpy().diagonal()

            for index, prob in enumerate(final_prob):
                if index == 0:
                    continue
                if index in target:
                    prob_label_result.append((prob, index, 1))
                else:
                    prob_label_result.append((prob, index, 0))

        # stop_time = time.time()
        # print 'The time to load test data: ', stop_time - start_time, '\n' 

        prob_label_result_sorted = sorted(prob_label_result, key=operator.itemgetter(0), reverse=True)

        TP = 0
        FP = 0
        all_pos = len([item for item in prob_label_result if item[2] == 1])

        precision = []
        recall = []
        f1 = []

        for (prob, label, result) in prob_label_result_sorted[:2000]:
            if result == 1:
                TP += 1
            else:
                FP += 1

            pre_value = float(TP)/(TP+FP)
            rec_value = float(TP)/all_pos
            if pre_value == 0 and rec_value == 0:
                f1_value = 0
            else:
                f1_value = 2 * (pre_value * rec_value) / (pre_value + rec_value)

            precision.append(pre_value)
            recall.append(rec_value)
            f1.append(f1_value)

#         plt.plot(recall, precision)
#         plt.axis([0, 0.4, 0, 1])

        return recall, precision
        # return metrics.auc(recall, precision)




