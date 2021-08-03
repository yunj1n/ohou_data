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

# # 크롤링한 데이터 전처리 ✏︎

# ## 데이터 속성
#
# - 'userId' : 사용자아이디
# - 'productId' : 상품코드
# - 'rating' : 별점 1.0 ~ 5.0
# - 'timestamp' : 리뷰생성일
# - /ohou_crawling/ohou_review_{productId}.csv : 크롤링 원본 데이터
# - /ohou_review_data/ohou_rating_data.csv : 정제된 데이터

import pandas as pd
import glob
import os
import re

# +
path_dir = './ohou_crawling/ohou_review_*.csv'

file_list = glob.glob(path_dir)
file_list_py = [file for file in file_list if file.endswith(".csv")]
file_list

# +
allData = []
for file in file_list:
    df = pd.read_csv(file)
    allData.append(df)
    
dataCombine = pd.concat(allData, axis=0, ignore_index=True)
dataCombine.to_csv('./ohou_review_data/ohou_정제된데이터.csv', index=False)
# -

df = pd.read_csv('./ohou_review_data/ohou_정제된데이터.csv')
df.head(3)

# 크롤링 데이터
df.shape # (1188252, 13)

df.drop_duplicates(keep='first', inplace=True) # 중복행 중에 첫행만 남기고 제거

df = df[df['writer_id'] != 'writer_id'] # 중복 헤더 제거

df.reset_index(inplace=True) # 인덱스 리셋

# 정리된 데이터
df.shape # (974939, 14)

# ## 필요한 컬럼 값만 추출

# +
# 리뷰 평점만 추출
reviews = df['review']
ratings = []
for i in reviews:
    sp = i.strip().split() 
    rexp = re.findall('(\d\.\d)', str(sp))
    if len(rexp) > 0:
        ratings.append(rexp[0].split('.')[0])
        
# 리뷰 상품 코드 추출
prod = df['production_information']
productId = []
for i in range(0,len(prod)):
    prod_id = prod[i].split(':')[1].split(',')[0].strip()
    productId.append(prod_id)
            
# 리뷰 생성일 추출
times = df['created_at']
timestamps = []
for time in times:
    sp = time.split('.')
    if len(sp) == 3:
        stamp = ''.join(sp)
        timestamps.append(stamp)
        
# 리뷰 작성자 아이디 추출
users = df['writer_id']
userId = []
for i in users:
    userId.append(i)
# -

len(userId), len(ratings), len(timestamps), len(productId)

# +
# 추출한 데이터로 DataFrame 새로 생성
raw_data = {
    'userId' : userId,
    'productId' : productId,
    'rating': ratings,
    'timestamp' : timestamps
}

new_df = pd.DataFrame(raw_data)
new_df.head(3)
# -

new_df.dtypes

# 타입변경 (timestampe는 무시해도 열이라서 컬럼변경X)
new_df = new_df.astype({'userId':int,
               'productId':int,
               'rating':float})
new_df.dtypes

# csv 새로 생성
new_df.to_csv('./ohou_review_data/ohou_rating_data.csv',index=False)

# 확인
pd.read_csv('./ohou_review_data/ohou_rating_data.csv')
