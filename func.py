import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from konlpy.tag import Okt
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
import string




def funcDF_info(df, columns=None):
    print(f'DF.info :\n{df.info()}\n\n')
    print(f'DF.describe :\n{df.describe()}\n\n')
    print(f'DF 결측치 파악 :\n{df.isnull().sum()}\n\n')

def funcDF_heatmap1(df):
    sns.heatmap(df)

def funcDF_heatmap2(df):
    sns.heatmap(df.corr())

def funcDF_pairplot(df):
    sns.pairplot(df)

def funcDF_boxplot(df, columns=None):
    # 이상치 박스플롯
    if columns is None:
        columns = df.columns
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(15, 5))
    for i, column in enumerate(columns):
        sns.boxplot(x=df[column], ax=axes[i])
        axes[i].set_title(column)
    plt.tight_layout()
    plt.show()
    if columns is None:
        columns = df.columns
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
        print(f'{column}컬럼] Q1 : {Q1:.3f}, Q3 : {Q3:.3f}, IQR : {IQR:.3f}, 사분위범위 : {lower_bound:.3f} ~ {upper_bound:.3f}')


def funcDF_histogram(df):
    num_columns = df.shape[1]
    num_rows = int(num_columns ** 0.5)
    num_cols = (num_columns // num_rows) + (1 if num_columns % num_rows != 0 else 0)
    

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10,6))
    for i, column in enumerate(df.columns):
        row = i // num_cols
        col = i % num_cols
        sns.histplot(df[column], ax=axes[row, col], kde=True)
        axes[row, col].set_title(column)

    for i in range(num_columns, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.show()

def funcDF_visualize(df, columns=None):
    funcDF_histogram(df)
    funcDF_heatmap1(df)
    funcDF_heatmap2(df)
    funcDF_pairplot(df)
    funcDF_boxplot(df, columns=None)

def funcDF_logAPPLY(df, columns=None):
    if columns is None:
        columns = df.columns
    for column in columns:
        df[column] = np.log1p(df[column])
    print('Log apply')
    return df


def funcNLP_Eng_tokenize_text(text):
    tokens = word_tokenize(text)
    print('retrun : 토큰')
    return tokens

def funcNLP_Eng_tokenize_sentences(text):
    sentences = sent_tokenize(text)
    print('retrun : 문장')
    return sentences

def funcNLP_Eng_remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    print('retrun : 토큰')
    return filtered_tokens

def funcNLP_Eng_stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    print('retrun : 어간')
    return stemmed_tokens

def funcNLP_Eng_preprocess_text(text):
    tokens = funcNLP_Eng_tokenize_text(text)
    filtered_tokens = funcNLP_Eng_tokenize_sentences(tokens)
    stemmed_tokens = funcNLP_remove_Eng_stopwords(filtered_tokens)
    print('retrun : 텍스트 - 토큰 - 어간')
    return stemmed_tokens

okt = Okt()

def funcNLP_okt_tokenize(text):
    tokens = okt.morphs(text)
    return tokens

def funcNLP_okt_pos_tagging(text):
    pos_tagged = okt.pos(text)
    return pos_tagged

def funcNLP_okt_noun_extractor(text):
    nouns = okt.nouns(text)
    return nouns

kiwi = Kiwi()

def funcNLP_kiwi_tokenize(text):
    tokens = kiwi.tokenize(text)
    return tokens

def funcNLP_kiwi_pos_tagging(text):
    pos_tagged = kiwi.analyze(text)
    return pos_tagged

def funcNLP_kiwi_noun_extractor(text):
    nouns = [word.value for word in kiwi.analyze(text) if word.tag == 'NNG' or word.tag == 'NNP']
    return nouns

def funcNLP_kiwi_stopword(text):
    stopwords = Stopwords()
    stopword = stopwords.filter(kiwi.tokenize(text))
    print('return : 불용어 필터 적용')
    return stopword

def funcNLP_punctuation(df,column):
    punctuation = string.punctuation
    df.column.replace(r'[{}]'.format(string.punctuation), '', regex=True, inplace=True)
    hanguel_pattern = "[^ㄱ-ㅎㅏ-ㅣ가-힣]"
    df.column = df.column.str.replace(hanguel_pattern, '', regex=True)
    print('return : "[^ㄱ-ㅎㅏ-ㅣ가-힣]"정규화')
    return df

def funcNLP_isull_sum(df,column):
    print(df.column.isnull().sum())

def funcNLP_otk_stopword(stopwords_path, vocab):
    with open(stopwords_path, encoding="utf-8") as f:
        stopwords = f.readlines()
    stopwords = [word.strip() for word in stopwords]
    for sword in stopwords:
        if sword in vocab.keys():
            vocab.pop(sword)
    print('return : vocab')
    return vocab

def funcNLP_otk_vocab(df, vocab):
    for idx in range(df.shape[0]):
        result = okt.morphs(df.iloc[idx][0])
        for word in result:
            if len(word) >=2:
                if vocab.get(word) != None:
                    vocab[word] +=1
                else:
                    vocab[word] =1
    sorted_vocab = sorted(vocab.items(), key= lambda x:x[1], reverse=True)
    print('return : sorted_vocab')
    return sorted_vocab

def funcNLP_otk_vocab2(vocab,n):
    vocab2 ={}
    for k, v in vocab.items():
        if v>n:
            vocab2[k] = v
    print('return : vocab2 - vocab 중 n개 이상 등장')
    return vocab2

def funcNLP_vocab2_visualize(vocab2, n):
    sorted_vocab2 = sorted(vocab2.items(), key=lambda x: x[1], reverse=True)

    keys = [k for k, v in sorted_vocab2]
    values = [v for k, v in sorted_vocab2]
    print('vocab2 중 상위 n개 시각화')
    plt.figure(figsize=(10, 6))
    plt.bar(keys[:n], values[:n])
    plt.xticks(rotation=90)
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.title('vocab2')
    plt.show()