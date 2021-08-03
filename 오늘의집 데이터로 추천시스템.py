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

# # 추천시스템 소개
#
# - 사용자가 제품에 대한 정보를 찾을 수 있도록 돕기 위해 개발된 권장 시스템
# - 추천 시스템은 사용자와 아이템 간의 유사성을 생성하고, 사용자-아이템 간의 유사성을 활용하여 추천

# # 추천시스템 (6가지의 추천 유형)
#
# - 인기도 기반 시스템 : 대부분의 사람들이 보고 구매한 항목을 추천하는 방식으로 작동하며 높은 평가를 받으나, 개인화된 추천은 아님
# - 분류 모델 기반 : 사용자의 특성을 이해하고 분류 알고리즘을 적용하여 사용자가 해당 제품에 관심이 있는 지 여부를 결정
# - 콘텐츠 기반 추천: 사용자 의견이 아닌 아이템 콘텐츠에 대한 정보를 기반으로 추천하며 주요 아이디어는 사용자가 항목을 좋아하면 '다른' 유사한 항목을 좋아할 것이라고 예측하는 것
# - 협업 필터링 : 사용자가 좋아하는 것과 비슷한 것을 좋아하고 취향이 비슷한 다른 사용자들이 좋아하는 것을 좋아한다는 가정에 기반 하며 두 가지 유형으로 나뉨
#
#       a) 사용자-사용자
#       b) 아이템-아이템
# - 하이브리드 접근 방식 : 협업 필터링, 콘텐츠 기반 필터링 및 기타 접근 방시을 결합한 것
# - 연관 규칙 마이닝 : 연관 규칙은 트랜잭션 전반에 걸쳐 동시 발생 패턴을 기반으로 항목 간의 관계를 캡처

# # 추천시스템 코드 ✏︎

# ## Top-N 추천 시스템

from collections import defaultdict
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise import accuracy
import pandas as pd
import numpy as np

# dataFrame 읽어오기
df = pd.read_csv('./ohou_review_data/ohou_rating_data.csv')
df.head()

df.shape

df = df.iloc[:974939,0:]

# timestamp 열 삭제하기
df = df.drop(['timestamp'], axis=1)
df.head(3)

# dataset 읽어오기
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df,reader)

data.raw_ratings[:10]

trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset) # 훈련


# Top-N Recommendation 
def get_top_n(predictions, n):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x:x[1], reverse=True) # 내림차순으로 정렬
        top_n[uid] = user_ratings[:n]
    
    return top_n


testset = trainset.build_anti_testset()

predictions = algo.test(testset)

# n = 1
top_n = get_top_n(predictions, 10)

for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])


# ## 정밀도와  재현율

def precision_recall_at_k(predictions, k, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, treu_r, est, _ in predictions:
        user_est_true[uid].append((est,treu_r))
        
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # 예상 값으로 사용자 평가 정렬
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # 관련 항목 수
        n_rel = sum((treu_r >= threshold) for (_, true_r) in user_ratings)
        
        # 상위 k의 추천 항목 수
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        
        # 상위 k의 관련 및 추천 항목 수
        n_rel_and_rec_k = sum(((true_r >= threshold) and 
                               (est >= threshold)) 
                               for (est, true_r) in user_ratings[:k])
        
        # precision@k : 관련성 있는 추천 항목의 비율
        precisions[uid] = n_rel_and_rec_k/n_rec_k if n_rec_k!=0 else 1
        
        # recall@k : 추천된 관련 항목의 비율
        recalls[uid] = n_rel_and_rec_k/n_rel if n_rel!=0 else 1
    
    return precisions,recalls


kf = KFold(n_splits=5)
algo = SVD()

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions,
                                                5,
                                                threshold=4)
    
    print(sum(prec for prec in precisions.values())/len(precisions))
    print(sum(rec for rec in recalls.values())/len(recalls))

# ## accuracy

accuracy.rmse(predictions, verbose=True)

# # 속성 정보
#
# - userId : 고유한 ID로 식별되는 모든 사용자
# - productId : 고유 ID로 식별되는 모든 제품
# - rating : 해당 사용자가 해당 상품에 대한 등급
# - timestamp : 평가 시간 (해당 열을 무시)

import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib
import scipy.sparse 
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings
warnings.simplefilter('ignore')
# %matplotlib inline

# 한글 폰트 사용을 위한 세팅
from matplotlib import font_manager, rc
font_path = '/Users/yunj1n/Library/Fonts/NanumGothic.otf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

ohou_df = pd.read_csv('./ohou_review_data/ohou_rating_data.csv')
ohou_df.head()

ohou_df.shape # (974939, 4)

ohou_df.dtypes

ohou_df.info()

# 평점 요약
ohou_df.describe()['rating']

print('최소 평점 : %d' %(ohou_df.rating.min()))
print('최대 평점 : %d' %(ohou_df.rating.max()))

# Na 값 찾기
ohou_df.isna().sum()

# ## 평점

# 평점 분포 확인
with sns.axes_style('white'):
    g = sns.factorplot('rating', data=ohou_df, aspect=2.0, kind='count')
    g.set_ylabels('총 평점 수')

#     ☞ 대부분의 사람들이 5점을 주었음

# ## 고유 사용자 및 상품 수

print('전체 데이터 ')
print('-'*50)
print('\n총 리뷰 수 :', ohou_df.shape[0])
print('총 사용자 수 :', len(np.unique(ohou_df.userId)))
print('총 상품 수 :', len(np.unique(ohou_df.productId)))

# ## 평점 분석

# timestamp 열 삭제하기
ohou_df.drop(['timestamp'], axis=1, inplace=True)

# 사용자가 평가한 평점 분석
no_rated_prod_per_user = ohou_df.groupby(by='userId')['rating'].count().sort_values(ascending=False)
no_rated_prod_per_user.head()

no_rated_prod_per_user.describe()

quantiles = no_rated_prod_per_user.quantile(np.arange(0, 1.01, 0.01),
                                            interpolation='higher')

# +
plt.figure(figsize=(10,10))
plt.title('분위수 및 해당 값')
quantiles.plot()

plt.scatter(x=quantiles.index[::5],
            y=quantiles.values[::5],
            c='orange',
            label='0.05 간격의 분위수')

plt.scatter(x=quantiles.index[::25],
            y=quantiles.values[::25],
            c='m',
            label='0.25 간격의 분위수')

plt.ylabel('사용자별 평점 수')
plt.xlabel('분위수에서의 가치')
plt.legend(loc='best')
plt.show()
# -

print('\n 사용자별 30개 이상의 평가된 제품 수 : {}\n'.format(sum(no_rated_prod_per_user >= 30)))

# ## 인기도 기반 추천
#
# - 인기도 기반 추천 시스템은 트렌드와 함께 작동
# - 기본적으로 현재 유행하는 아이템을 사용
# - 모든 신규 사용자가 일반적으로 구매하는 상품을 방금 가입한 사용자에게 해당 항목을 제안할 가능성이 있음
# - 인기도 기반 추천 시스템의 문제점은 이 방법으로는 개인화를 사용할 수 없다는 것
# - 용자의 행동을 알더라도 그에 따라 항목을 추천할 수 없음

# 30개 이상의 평점을 부여한 사용자를 포함하는 새 데이터프레임 가져오기
new_df = ohou_df.groupby('productId').filter(lambda x:x['rating'].count() >= 30)

# +
no_of_rating_per_prod = new_df.groupby(by='productId')['rating'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_rating_per_prod.values)
plt.title('상품별 평점')
plt.xlabel('상품')
plt.ylabel('상품별 평점 수')
ax.set_xticklabels([])

plt.show()
# -

# 상품의 평균 평점
new_df.groupby('productId')['rating'].mean().head()

new_df.groupby('productId')['rating'].mean().sort_values(ascending=False)

# 상품에 대한 총 평점 수
new_df.groupby('productId')['rating'].count().sort_values(ascending=False)

ratings_mean_cnt = pd.DataFrame(new_df.groupby('productId')['rating'].mean())
ratings_mean_cnt['rating_counts'] = pd.DataFrame(new_df.groupby('productId')['rating'].count())
ratings_mean_cnt.head()

ratings_mean_cnt['rating_counts'].max()

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_cnt['rating_counts'].hist(bins=50)

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_cnt['rating'].hist(bins=50)

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='rating', y='rating_counts',
              data=ratings_mean_cnt, alpha=0.4)

popular_products = pd.DataFrame(new_df.groupby('productId')['rating'].count())
most_popular = popular_products.sort_values('rating', ascending=False)
most_popular.head(30).plot(kind='bar')

# ## 협업 필터링 (CF)
#
# - 협업 필터링은 일반적으로 추천 시스템에 사용됨 이러한 기술은 사용자 항목 연관 매트릭스의 누락된 항목을 채우는 것을 목표로함
# - 협업 필터링은 비슷한 취향을 가진 사람들에게서 최고의 추천이 나온다는 생각에서 출발함
# - 비슷한 생각을 가진 사람들의 과거 항목 평가를 사용하여, 누군가가 항목을 어떻게 평가할지 예측
# - 협업 필터링에는 일반적으로 메모리 기반 접근 방식과 모델 기반 접근 방식이라고 하는 두가지 하위 범주가 있음

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

# dataset 읽어오기
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(new_df, reader)

# dataset 분할하기
trainset, testset = train_test_split(data, test_size=0.3, random_state=10)

# 'user_based' true/false 를 사용하여 사용자 기반 또는 아이템 기반 협업 필터링간 전환
algo = KNNWithMeans(k=5, sim_options={
                                    'name' : 'pearson_baseline',
                                    'user_based' : False})
algo.fit(trainset)

# 테스트 세트에 대해 훈련된 모델 실행
test_pred = algo.test(testset)
test_pred

# RMSE 구하기
print('아이템 기반 모델 : 테스트 세트')
accuracy.rmse(test_pred, verbose=True)

# ## 모델 기반 협업 필터링 시스템
#
# - 이 방법은 기계 학습 및 데이터 마이닝 기술을 기반으로 함
# - 목표는 예측을 할 수 있도록 모델을 훈련시키는 것
# - 예를 들어, 기존 사용자-아이템 상호 작용을 사용하여 사용자가 가장 좋아할 것 같은 상위 5개 항목을 예측하도록 모델을 훈련할 수 있음
# - 이러한 방법의 장점 중 하나는 메모리 기반 접근 방식과 같은 다른 방법에 비해 더 많은 수의 사용자에게 더 많은 항목을 추천할 수 있다는 것
# - 큰 희소 행렬로 작업할 때도 적용 범위가 넓음

new_df.shape

new_df1 = new_df.head(100000)
ratings_matrix = new_df1.pivot_table(values='rating', 
                                     index='userId',
                                     columns='productId',
                                     fill_value=0)
ratings_matrix.head()

ratings_matrix.shape

# 행렬 전치
X = ratings_matrix.T
X.head()

X.shape

# 데이터 하위 집합의 고유 제품
X1 = X

# +
from sklearn.decomposition import TruncatedSVD

# 행렬 분해
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape
# -

# 상관 행렬
correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape

X.index[32]

# +
i = 832028 # 사용자가 구매한 상품ID 코드값

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID
# -

# 동일한 상품을 구매한 다른 사용자가 평가한 아이템을 기반으로 
# 사용자가 구매한 아이템에 대한 상관 관계 계산
correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID

# 상관관계가 높은 상품을 순서대로 추천
Recommend = list(X.index[correlation_product_ID > 0.6])
Recommend.remove(i) # 사용자가 이미 구매한 아이템은 제거
Recommend[0:5]

# ## '832028' 와 상관계수가 높은 상품 추천 리스트

# ### 17087 [베베데코] 빠른배송 3중직 암막커튼 10color
# ### 평점 4.6
#
# ![17087](https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/162547032356158411.jpg?gif=1&w=1024&h=1024&c=c&webp=1)

# ### 335355 [비알프렌드] 디디테이블 반타원형 라미네이트 화이트식탁(6size)
# ### 평점 4.7
#
# ![335355](https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/162376965287757163.jpeg?gif=1&w=1024&h=1024&c=c)

# ### 355500 [모던바로크] 비앙카 LED 수납 침대 (매트포함) SS/Q/K
# ### 평점 4.6
#
# ![355500](https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/162158717957439878.jpg?gif=1&w=1024&h=1024&c=c)

# ### 365562 [보니애가구] 조안 아쿠아텍스 생활방수 2인용 패브릭소파 3colors
# ### 평점 4.8
#
# ![365562](https://image.ohou.se/i/bucketplace-v2-development/uploads/productions/159834077410020432.jpg?gif=1&w=1024&h=1024&c=c)
