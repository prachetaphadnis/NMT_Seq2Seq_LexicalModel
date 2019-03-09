import pickle
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torch.serialization import default_restore_location

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler

from matplotlib.font_manager import FontProperties
from matplotlib import rcParams

def get_args():
	""" Defines training-specific hyper-parameters. """
	parser = argparse.ArgumentParser('Sequence to Sequence Model')
	parser.add_argument('--cuda', default=False, help='Use a GPU')
	# Add data arguments
	parser.add_argument('--data', default='prepared_data', help='path to data directory')
	parser.add_argument('--source-lang', default='jp', help='source language')  
	return parser.parse_args()
def main(args):
	file = open("prepared_data/test.jp",'rb')
	object_file = pickle.load(file)
	file.close()
	#src_str = src_dict.string(src_ids).split(' ')
	src_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.source_lang)))
	#print('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
	src_str=src_dict.string(object_file[30]).split(' ')
	print(src_str)

if __name__ == '__main__':
	args = get_args()
	main(args)