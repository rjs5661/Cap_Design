#!/usr/bin/env python
# coding: utf-8

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from wordcloud import WordCloud

WRKON_JUPYTER = os.path.exists('../../resources')

if WRKON_JUPYTER:
    sys.path.append(os.path.abspath('../..'))

else:
    sys.path.append(os.path.abspath('..'))

import resources



def plot_freq_pie(top_ns, ax):
    ax.pie(top_ns.values, labels=top_ns.index, autopct='%.1f%%', startangle=90)
    #ax.axis('equal')
    ax.set_title('뉴스 제목 단어 빈도')
    ax.legend(loc = (1, 0.6), title = '빈도 수')

def plot_tag_cnt_pie(tag_cnts, ax):
    ax.pie(tag_cnts.values(), labels=tag_cnts.keys(), autopct='%1.1f%%', startangle=90)
    ax.set_title('카테고리별 기사 개수')

def plot_freq_bar(top_ns, ax):
    ax.bar(range(top_ns.size), top_ns.values)
    ax.set_xticks(range(top_ns.size), top_ns.index, rotation=45)

def plot_wc(top_ns, ax):
    wc = WordCloud(
        font_path=resources.font_path,
        width=int(ax.bbox.width), height=int(ax.bbox.height)
    )
    cloud = wc.generate_from_frequencies(top_ns)

    ax.imshow(wc)
    ax.axis('off')

def plot_summary(word_changes, start, end, ax, max_token_show=10):

    prompt = f'{start}부터 {end}까지의 분석 요청 결과입니다\n'
    prompt += '해당 기간 동안 꾸준한 관심을 받은 뉴스 기사는 '
    for tk in word_changes['steady'][ : max_token_show]:
        prompt += f'{tk}, '

    prompt = prompt[ : -2]
    prompt += '에 관한 기사입니다.\n'
    
    prompt += '해당 기간 동안 '
    for tk in word_changes['increase'][ : max_token_show]:
        prompt += f'{tk}, '

    prompt = prompt[ : -2]
    prompt += '에 관한 기사에 대한 관심은 줄어드는 추세를 보였습니다.\n'

    prompt += '반면 '
    for tk in word_changes['decrease'][ : max_token_show]:
        prompt += f'{tk}, '

    prompt = prompt[ : -2]
    prompt += '에 관한 기사들은 새로 관심을 받기 시작하였습니다.'

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    
    ax.text(0.5 * (left + right), 0.5 * (bottom + top), prompt,
        horizontalalignment='center',
        verticalalignment='center'
           )
    ax.axis('off')
    
def plot_hm(top_ns, ax):
    sns.heatmap(top_ns, ax=ax)
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())



def plot_board(tag_cnts, top_ns, word_changes, start, end):
    
    fig, axes = plt.subplot_mosaic(
        [
            ['pie_freq', 'pie_tag_cnt', 'bar_cnt'],
            ['wc', 'txt_summary', 'txt_summary'],
            ['hm', 'hm', 'hm']
        ],
        figsize=(25, 10)
    )
    
    plot_freq_pie(top_ns.sum().iloc[ : 10], axes['pie_freq'])
    plot_tag_cnt_pie(tag_cnts, axes['pie_tag_cnt'])
    plot_freq_bar(top_ns.sum(), axes['bar_cnt'])

    plot_wc(top_ns.sum(), axes['wc'])
    plot_summary(word_changes, start, end, axes['txt_summary'])
    
    plot_hm(top_ns, axes['hm'])

    return fig


# hm_data = melten_titles.pivot_table(
#     values='counts',
#     index='dates',
#     columns='tokens',
#     aggfunc='sum',
#     fill_value=0
# )
# 
# hm_tokens = hm_data.sum().sort_values(ascending=False).iloc[:100].index

# fig = plt.figure(figsize=(25, 8))
# 
# ax = sns.heatmap(
#     hm_data[hm_tokens]
# )
# 
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15)
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
# 
# plt.savefig('tst_hm.png')
# 
# plt.show()
