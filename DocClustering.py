# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from bs4 import BeautifulSoup
import mpld3

""" ----------- 第一步：导入相关文档 -------------"""

titles = open('title_list.txt').read().split('\n')

synopses_wiki = open('synopses_list_wiki.txt', encoding='UTF-8').read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]
synopses_clean_wiki = []
for text in synopses_wiki:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean_wiki.append(text)
synopses_wiki = synopses_clean_wiki

synopses_imdb = open('synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]
synopses_clean_imdb = []
for text in synopses_imdb:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean_imdb.append(text)
synopses_imdb = synopses_clean_imdb

synopses = []
for i in range(len(synopses_wiki)):
    item = synopses_wiki[i] + synopses_imdb[i]
    synopses.append(item)


""" ----------- 第二步：分词和词干提取 -------------"""


stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
# 分词+提取词干
def tokenize_and_stem(text):
    # 先将文章按句子进行分割，然后句子进行分词
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤掉不包含字幕的单词
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

# 仅分词
def tokenize_only(text):
    # 先将文章按句子进行分割，然后句子进行分词
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤掉不包含字幕的单词
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list   
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)


""" ----------- 第三步：Tf-idf和文档相似性 -------------"""


from sklearn.feature_extraction.text import TfidfVectorizer
#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
"""
max_df：这个给定特征可以应用在 tf-idf 矩阵中，用以描述单词在文档中的最高出现率。
        假设一个词（term）在 80% 的文档中都出现过了，那它也许（在剧情简介的语境里）
        只携带非常少信息。
min_df：可以是一个整数（例如5）。意味着单词必须在 5 个以上的文档中出现才会被纳入考虑。
        设置为 0.2；即单词至少在 20% 的文档中出现。
ngram_range：这个参数将用来观察一元模型(unigrams)，二元模型(bigrams)
         和三元模型(trigrams)。参考n元模型（n-grams）。
max_features：最多的特征个数(每个单词为一个特征)
"""

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses
print(tfidf_matrix.shape)

# 提取的单词（在20%-80%的文档中出现的term）
terms = tfidf_vectorizer.get_feature_names()
# 没有被选中的term，包括：
# （1） occurred in too many documents (max_df)
# （2） occurred in too few documents (min_df)
# （3） were cut off by feature selection (max_features).
stop_words_ = tfidf_vectorizer.stop_words_ 

# 测量任意两个概要之间的余弦距离
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)











