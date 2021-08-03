# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 데이터 크롤링
#
# ## 어떤 데이터를 수집할 것인가?
#
# - 추천시스템 구현을 위한 리뷰 데이터를 수집
# - [오늘의집](https://ohou.se/)
# - 오늘의집은 정보 탐색부터 제품 구매까지 인테리어의 모든 과정을 한 플랫폼에서 제공하는 서비스로 2020년 10월 기준 1,400만 다운로드, 누적 회원 수가 1,000만 명이며 사용자들이 올린 컨텐츠 수가 750만이고 월 거래액이 700억임

# ### 오늘의집 > 스토어 > 베스트 > 역대 베스트 > 가구,패브릭,홈데코 등 카테고리에서 상품 추출

# ![베스트상품](./img/ohou1.png)

# ## 데이터 수집
#
# - product_id : 상품코드
# - page : 페이지 (리뷰 한페이지에 5개씩 데이터 노출)
# - order : 베스트순 - best / 최신순 - recent / 낮은 평점순 - worst

# #!pip install requests
import os
import re
import sys
import urllib.request
import time
import json
import csv
import requests
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# 오늘의 집 베스트랭킹 상품 코드/리뷰 수 추출
def get_prod_id(_type,category,page):
    url = 'https://ohou.se/commerces/ranks.json'
    value = {
        'v' : 2,
        'tpye' : _type,
        'category' : category,
        'page' : page,
        'per' : 20
    }
    params = urllib.parse.urlencode(value)
    
    req = requests.get(url, params=params)
    status = req.status_code
    
    id = []
    cnt = []
    result = []
    
    if status == 200:
        data = req.json()
        for i in range(0,20):
            id.append(data['products'][i]['id'])
            cnt.append(data['products'][i]['review_count'])
            
    else:
        print(status)
        return
    
    result.append(id)
    result.append(cnt)
    
    return result


ranks=[]
for page in range(1,6):
    ranks.append(get_prod_id('best',4,page))


# 오늘의 집 리뷰 json 호출
def ohou_review_json(production_id, page, order):
    url = 'https://ohou.se/production_reviews.json'
    value = {
        'production_id' : production_id,
        'page' : page ,
        'order' : order
    }
    params = urllib.parse.urlencode(value)
    
    req = requests.get(url, params=params)
    status = req.status_code
    
    if status == 200:
        data = req.json()
        ohou_review_to_csv(data)
    else:
        print(status)
        return


# 오늘의 집 리뷰 json => csv
def ohou_review_to_csv(data):
    info = data['reviews']
    # print(info)
    
    with open('./ohou_crawling/ohou_review_{}.csv'.format(production_id), 'a') as f:
        cw = csv.DictWriter(f, fieldnames=info[0].keys())
        cw.writeheader()
        cw.writerows(info)


# +
# 호출
production_id = ranks[1][0][13]
review_cnt = ranks[1][1][13]
print(production_id, review_cnt)

for page in range(1,round((review_cnt/5)+1)):
    ohou_review_json(production_id, page, 'best')
# -

# ## 수집한 데이터 저장
#
# - /ohou_crawling/ohou_review_{productId}.csv : 크롤링 원본 데이터

# +
# 크롤링 데이터 불러오기 
df = pd.read_csv('./ohou_crawling/ohou_review_{}.csv'.format(production_id))

df.head()
