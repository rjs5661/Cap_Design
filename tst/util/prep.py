#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
from collections import Counter
from konlpy import tag
import nltk
import numpy as np
import os
import pandas as pd
import re
import requests
import spacy
import sys
import textacy.preprocessing as tprep

WRKON_JUPYTER = os.path.exists('../../resources')

if WRKON_JUPYTER:
    sys.path.append(os.path.abspath('../..'))

else:
    sys.path.append(os.path.abspath('..'))
    
import resources

def get_punct_list(title):
    return re.findall(r'[^ㄱ-ㅎ-가-힣\w\s\(\{\[\)\}\]]', title)

def get_punct_set(titles):
    punct_set = set()
    titles.apply(lambda x: punct_set.update(get_punct_list(x)))
    return punct_set

def get_punct_freq(titles):
    punct_set = get_punct_set(titles)
    punct_freq = {p : 0 for p in punct_set}
    for t in titles:
        for p in get_punct_list(t):
            punct_freq[p] += 1

    return punct_freq


# In[ ]:


# 제목의 불순도(문장 길이 대비문장 부호 처리 결과 품질 측정 목적
# 변경 사항 (050224 1920)
# 1 .첫 줄 삭제 : cpy = title[:]
# 2. (괄호 표현) 감지 부분 변경 : . -> _ , cpy -> title
def get_impurity_score(title:str):
    cpy = re.sub(r'[\(\{\[]+[ㄱ-ㅎ-가-힣\w\s,]+[^ㄱ-ㅎ-가-힣\w\s]*[\]\}\)]+', '_', title)
    cpy = re.sub(r'\s', '', cpy)

    n_chars = len(cpy) if len(cpy) != 0 else 1 # (copyright) 같은 제목은 처리 후엔 길이 0 -> 1로 간주
    n_puncts = len(get_punct_list(cpy))
    
    return round(n_puncts / n_chars, 3)


# In[30]:


def replace_sokbo_into_ub(title):
    return re.sub(r'[\(\{\[]+[ㄱ-ㅎ-가-힣\w\s,]+[^ㄱ-ㅎ-가-힣\w\s]*[\]\}\)]+', '_', title)


# In[31]:


def normalize_punct(title):

    punct_list = ['\'', '\"', '…', ',', '‥', '!', '@', '#', '&', '/', '+', '-', '=', '~', '?', '>', '_', '㈜', '↓', '↑', '→']
    
    title = replace_sokbo_into_ub(title) # (속보), [단독] 따위의 [000의 건강상식]과 같은 요소들은 _으로 변경
    title = title.translate(resources.TRANSLATE_TABLE)
    
    title = tprep.normalize.quotation_marks(title) # 따옴표 정규화
    
    title = re.sub(r'\.\.(\.)?', '…', title) # 말줄임표 정규화 ('..' , '...' -> '…')
    
    title = tprep.normalize.bullet_points(title)
    title = re.sub(r'·', ' ', title) # 불릿 표현 정규화 + 띄어쓰기로 변형 -> 추후 품사 태깅 등을 통해 낱말 조합 등 진행

    title = tprep.remove.punctuation(title, only=punct_list)
    
    title = re.sub('\s+', ' ', title) # 위에서 생긴 연속 공백 제거
    title = title.strip() # 양 끝 공백 제거
    
    return title


# In[ ]:


def custom_tokenize(title, kr_module=resources.DEFAULT_KR_TOKENIZER):

    josa_tag = None

    if isinstance(kr_module, tag.Okt): 
        josa_tag = ['Josa']
    elif isinstance(kr_module, tag.Komoran):
        josa_tag = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JC', 'JX']
    
    title = kr_module.pos(title)
    title = [t for t in title if t[1] not in josa_tag]
    title = [t[0] for t in title if t[0] not in resources.DEFAULT_STOPWORDS]
    title = ' '.join(title)

    return title



def melt_record(index, record):
    melten = Counter(record['titles'].split(' '))
    melten = pd.DataFrame.from_dict(melten, orient='index').T
    melten['index'] = index
    melten['dates'] = record['dates']
    melten = melten.melt(id_vars=['index', 'dates'], var_name='tokens', value_name='counts')
    melten = melten.rename(columns={'variable': 'tokens', 'value': 'counts'})

    return melten



def melt_titles(titles):
    melten_records = []

    print(f'melt : {len(titles)} records')
    for i, record in titles.iterrows():
        melten_records.append(melt_record(i, record))
        if i % 10000 == 0 : print('\nnow on record', end='\t')
        if i % 1000 == 0 : print(i, end='\t')

    print('concat records...')
    melten_records = pd.concat(melten_records)

    return melten_records



def get_cntvec(melten):
    cntvec = melten.pivot_table(
        values='counts',
        index='dates',
        columns='tokens', 
        aggfunc='sum', fill_value=0
    )
    return cntvec



def get_top_n_tokens(cntvec, n_tokens=100):
    top_n_tokens = cntvec.sum().sort_values(ascending=False).iloc[ : n_tokens].index

    return top_n_tokens



def tst_pt(melten):
    cntvec = melten.pivot_table(
        values='counts',
        index='tokens', ###
        columns='dates', ###
        aggfunc='sum', fill_value=0
    )
    return cntvec

def get_top_n_tokens_from_date(cntvec, date, n_tokens=100):
    top_n_tokens = cntvec[date].sort_values(ascending=False).iloc[ : n_tokens].index

    return top_n_tokens


def get_top_n_tokens_from_period(cntvec, n_tokens=100):
    top_n_tokens = []
    for date in cntvec.columns:
        date_top_n_tokens = get_top_n_tokens_from_date(cntvec, date)
        date_top_n_tokens = set(date_top_n_tokens)
        top_n_tokens.append(date_top_n_tokens)
    return top_n_tokens

def analyze_word_frequency_change(top_n_tokens): #단어 증가 감소 유지 함수
    word_changes = {'increase': [], 'decrease': [], 'steady': []}#증가, 감소, 유지 딕셔너리
    old_top100 = top_n_tokens[0]# 첫 번째 날짜의 단어 빈도를 가져오기.
    for top100 in top_n_tokens[1:]:# 두 번째 날짜부터 끝 날짜의 단어 빈도 가져와서 연산
        increase = list(top100 - old_top100) # 빈도가 증가한 단어를 찾기
        decrease = list(old_top100 - top100) # 빈도가 감소한 단어를 찾기
        steady = list(top100 & old_top100) # 빈도 유지한 단어를 찾기
              
        #각 단어들 딕셔너리에 추가
        word_changes['increase'].extend(increase)
        word_changes['decrease'].extend(decrease)
        word_changes['steady'].extend(steady)
        
        old_top100 = top100 # 이전 단어 빈도를 현재로 업데이트
        
    return word_changes