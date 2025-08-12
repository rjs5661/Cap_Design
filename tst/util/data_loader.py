#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import os
import pandas as pd
import pickle
import re
import requests
from sys import stdout
import time

WRKON_JUPYTER = os.path.exists(os.path.abspath('../../resources'))

def get_user_input():
    start = input('분석 시작 지점을 예시와 같이 입력하시오 (ex. 2024년 1월 1일 -> 20240101)')
    end = input('분석 종료 지점을 예시와 같이 입력하시오 (ex. 2024년 1월 1일 -> 20240101)')

    print(
        f"\n\t분석 시작일 : {start[:4]}년 {start[4:6]}월 {start[6:]}일\n\t분석 종료일 : {end[:4]}년 {end[4:6]}월 {end[6:]}일\n"
    )
    is_proceed = input('맞으면 y, 틀리면 n을 입력하시오([y]/n)')
    
    if (is_proceed == 'y') or (is_proceed == '') :
        return start, end
    else:
        print("분석을 종료합니다\n\n")
        return None, None



def load_raw_data(start, end):
    
    fpath = '../../data/' if WRKON_JUPYTER else '../data/'
    
    raw_data_list = []
    
    period = pd.date_range(start, end).strftime('%Y%m%d')

    for date in period:
        raw_data = pd.read_csv(
            fpath+date+'.csv'
        )
        raw_data['dates'] = date
        raw_data_list.append(raw_data)

    df = pd.concat(raw_data_list).reset_index(drop=True)

    return df



def get_title_cnt(date, sid1='001'):

    BASE_URL = f'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1={sid1}&listType=title&'

    last_page = 1000
    n_titles = 0

    while True:
        url = BASE_URL
        url += f'date={date}&'
        url += f'page={last_page}'

        res = requests.get(url)
        bs = BeautifulSoup(res.text, features='html.parser')
        time.sleep(0.1)

        has_next = bs.find('a', class_='next nclicks(fls.page)')
        page_list = bs.find('div', class_='paging')
        
        if not has_next and last_page >= int(page_list.find('strong').get_text()):
            last_page = int(page_list.find('strong').get_text())
            n_titles = len(bs.find_all('a', class_="nclicks(fls.list)"))
            break
        else :
            last_page += 1000
    
    return (last_page - 1) * 50 + n_titles



def get_title_cnt_in_period(date_start, date_end, sid1='001'):
    
    title_cnt = pd.date_range(date_start, date_end).strftime("%Y%m%d").to_series()
    title_cnt = title_cnt.apply(get_title_cnt, sid1=sid1)
    title_cnt = title_cnt.sum()

    return title_cnt



def crawl(start, end, time_sleep=0.5, page_start=1) :
    BASE_URL = 'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&listType=title&'

    period = pd.date_range(start, end).strftime('%Y%m%d')
    prompt = ''
    
    for date in period:
        
        raw_data = []
        max_page = get_title_cnt(date) // 50 + 1
        page = page_start
        pct = 0
        crawl_start = datetime.now()
        
        while True:            
            url = BASE_URL
            url += f'date={date}&'
            url += f'page={page}'
    
            res = requests.get(url)
            bs = BeautifulSoup(res.text, features='html.parser')
            time.sleep(time_sleep)

            raw_titles = [e.get_text() for e in bs.find_all('a', class_="nclicks(fls.list)")]
            raw_data.extend(raw_titles)

            pct = int(page * 100 / max_page)
            prompt = f'\r{date} ( MAX PAGE : {max_page} ) : '
            prompt += f'{page}, {pct}%'
            stdout.write(prompt)

            has_next = bs.find('a', class_='next nclicks(fls.page)')
            page_list = bs.find('div', class_='paging')
            
            if not has_next and int(page_list.find('strong').get_text()) >= max_page:
                break
            else : 
                page += 1

        prompt = '\tDONE '
        prompt += f'(start: {crawl_start}, '
        prompt += f'end: {datetime.now()}, '
        prompt += f'time: {datetime.now() - crawl_start}) / '
        
        stdout.write(prompt)

        if WRKON_JUPYTER : 
            save_raw_data(pd.Series(raw_data, name='titles'), fname=date+'.csv', fpath='../../data/')
        else : 
            save_raw_data(pd.Series(raw_data, name='titles'), fname=date+'.csv', fpath='../data/')

    return True



def save_raw_data(raw_data, fname, fpath):
    
    print('saving... ', end='')
    raw_data.to_csv(fpath+fname, index=False)
    print('DONE')

    return True



def convert_pkl_to_csv(fname):
    fpath = ''
    if WRKON_JUPYTER : 
        fpath='../../data/'
    else : 
        fpath='../data/'

    data = pd.read_pickle(fpath + fname)
    data = pd.Series(data['titles'], name='titles')
    data.to_csv(fpath+fname[:-4]+'.csv', index=False)
    return True




