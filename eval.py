# -*- coding: utf-8 -*-

import os
import math
import gensim
import time
import sys
import json
import copy
import operator
import argparse
import utils

import numpy as np
from scipy.misc import logsumexp
from random import shuffle

from scipy.sparse import hstack, vstack
from collections import defaultdict, Counter
from gensim.models import word2vec
from sklearn import metrics

from utils import loader, helper
from model.PCNN_ATT import PCNN_ATT


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd



def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--model_dir', type=str, default='saved_models/', help='Directory of the model.')
	parser.add_argument('--model', type=str, default='best_model.tar', help='Name of the model file.')
	parser.add_argument('--data_dir', type=str, default='data/')
	parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")

	parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
	parser.add_argument('--cpu', action='store_true')
	args = parser.parse_args()

	if args.cpu:
	    args.cuda = False
	    
	# make opt
	opt = vars(args)

	opt['train_file'] = opt['data_dir'] + '/' + 'train.txt'
    opt['test_file'] = opt['data_dir'] + '/' + 'test.txt'
    opt['vocab_file'] = opt['data_dir'] + '/' + 'vec.bin'
    opt['rel_file'] = opt['data_dir'] + '/' + 'relation2id.txt'

	# Pretrained word embedding
	print "Load pretrained word embedding"
	w2v_model = gensim.models.KeyedVectors.load_word2vec_format(opt['vocab_file'], binary=True)
	word_list = [u'UNK'] + w2v_model.index2word
	word_vec = w2v_model.syn0

	word_map = {}

	for id, word in enumerate(word_list):
	    word_map[word] = id

	assert opt['emb_dim'] == w2v_model.syn0.shape[1]


	# Read from relation2id.txt to build a dictionary: rel_map
	rel_map = {}
	        
	with open(opt['rel_file'],'rb') as f:
	    for item in f:
	        [relation, id] = item.strip('\n').split(' ')
	        rel_map[relation] = int(id)

	opt['num_rel'] = len(rel_map)
	opt['vocab_size'] = len(word_list)


	# Load data
	all_data = loader.DataLoader(opt['train_file'], opt['test_file'], opt, word_map, rel_map)
	opt['pos_e1_size'] = all_data.pos_max_e1 - all_data.pos_min_e1 + 1
	opt['pos_e2_size'] = all_data.pos_max_e2 - all_data.pos_min_e2 + 1
	opt['pos_min_e1'] = all_data.pos_min_e1
	opt['pos_min_e2'] = all_data.pos_min_e2

	assert opt['pos_e1_size'] == opt['pos_e2_size']

	helper.print_config(opt)

	model_file = args.model_dir + '/' + args.model

	# Load input checkpoint

	PCNN_ATT_model = PCNN_ATT(word_vec, opt)
	checkpoint = torch.load(model_file)
	PCNN_ATT_model.load_state_dict(checkpoint['state_dict'])

	PCNN_ATT_model.cuda()

	recall, precision = PCNN_ATT_model.test(all_data)
	test_AUC = metrics.auc(recall, precision)

	print 'the AUC of P/R curve of held-out evaluation is: %f' % test_AUC

	if len(args.out):

		helper.check_dir(os.path.dirname(args.out))
		pickle.dump(list(zip(precision, recall)), open(args.out, "w"))

		print "Predicted precision/recall save to %s" % args.out



	print "Evaluation ended."


if __name__ == "__main__":
    main()










