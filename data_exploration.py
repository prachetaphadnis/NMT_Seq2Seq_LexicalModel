import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_set):
	data_path=os.path.join(os.getcwd(),'raw_data', data_set)
	file = open(data_path,"r")
	train_set={}
	i=0
	for line in file:
		cur_line=line.split()
		train_set[i]=[cur_line]
		train_set[i].append(len(cur_line))
		i=i+1
	return train_set

def plot_distributions(data_set1, data_set2,data_set_desc1, data_set_desc2):
	sns.distplot([data_set1[i][1] for i in range(len(data_set1))], kde=False, label=data_set_desc1)
	sns.distplot([data_set2[i][1] for i in range(len(data_set2))], kde=False, label=data_set_desc2)
	#sns.distplot([data_set3[i][1] for i in range(len(data_set3))], kde=False, label=data_set_desc3)
	plt.legend()
	plt.xlabel("Sentence Length")
	plt.ylabel("Frequency")
	plt.title("Distribution of sentence lengths in English and Japanese training data")
	plt.show()

def plot_correlations(data_set1,data_set2):
	ax=sns.scatterplot(x=[data_set1[i][1] for i in range(len(data_set1))], y=[data_set2[i][1] for i in range(len(data_set2))] )
	plt.xlabel("English")
	plt.ylabel("Japanese")
	plt.title("Correlation between sentence lengths")
	plt.show()

def compute_types_and_tokens(data_set):
	word_dict={}
	for i in range(len(data_set)):
		for j in data_set[i][0]:
			if j in word_dict:
				word_dict[j]+=1
			else:
				word_dict[j]=1
	unk=0
	for val in word_dict.values():
		if val==1:
			unk+=1
	#print(word_dict['japanese'])

	token_list=[]
	for key,val in word_dict.items():
		if val==1:
			token_list.append(key)
	#print(token_list)

	return np.sum(list(word_dict.values())),len(word_dict.keys()),unk


if __name__ == '__main__':
	train_en=load_data('train.en')
	train_jp=load_data('train.jp')
	plot_distributions(train_en, train_jp, "English", "Japanese")
	#plot_distributions(train_jp,"Japanese data")
	plot_correlations(train_en,train_jp)
	num_word_tokens,num_word_types, unk=compute_types_and_tokens(train_en)
	print("Number of word tokens in English are:",num_word_tokens,"\nNumber of word types in English are:",num_word_types,"\nNumber of words that will be replaced by <UNK> in English:", unk)
	num_word_tokens,num_word_types,unk=compute_types_and_tokens(train_jp)
	print("Number of word tokens in Japanese are:",num_word_tokens,"\nNumber of word types in Japanese are:",num_word_types,"\nNumber of words that will be replaced by <UNK> in Japanese:", unk)
	#print(compute_types_and_tokens(train_en))
	