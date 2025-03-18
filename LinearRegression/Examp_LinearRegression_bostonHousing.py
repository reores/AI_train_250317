#!/usr/bin/env python
# coding: utf-8

# In[33]:


#선형회귀 문제
#Boston Housing : 주택 환경 조건에 따른 집 값 예측 모델
#아래 코드는 룸 개수에 따른 집 값 예측 모델을 구성하였음
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#훈련데이터 404개, 테스트 데이터 102개
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_train[0])
print(x_test[0])
print(y_train[0])
print(y_test[0])


# In[18]:


# 0. CRIM     per capita crime rate by town 마을별 1인당 범죄율
# 1. ZN       proportion of residential land zoned for lots over 25,000 sq.ft. 주거용 토지비율
# 2. INDUS    proportion of non-retail business acres per town 비소매시설(회사) 비율
# 3. CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 강가 1, 아니면 0
# 4. NOX      nitric oxides concentration (parts per 10 million) 일산화질소 농도 천만분의 1
# 5. RM       average number of rooms per dwelling 평균 객실 수
# 6. AGE      proportion of owner-occupied units built prior to 1940 주택년한(1940년 이전 노후주택)
# 7. DIS      weighted distances to five Boston employment centres 고용센터 5곳까지의 가중 거리
# 8. RAD      index of accessibility to radial highways 고속도로 접근성 지수
# 9. TAX      full-value property-tax rate per $10,000 재산세율
# 10. PTRATIO  pupil-teacher ratio by town 학생, 교사 비율
# 11. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 흑인 비율
# 12. LSTAT    % lower status of the population 인구 밀집도
# 13. MEDV     Median value of owner-occupied homes in $1000's 주택 가격 중앙값(단위 : 천 달러)
print(x_train[0])


# In[19]:


plt.figure(figsize=(7,6))
for ix in range(len(x_train[0])) :
    plt.subplot(5, 3, ix + 1) #3행 5열
    plt.scatter(x_train[:,ix], y_train, s=3) # x축, y축, 포인트 size
    plt.title(f"[{ix}]")
plt.show()


# In[20]:


# index 5(평균 객실 수), 12(인구 밀집도) 선형성 확인
# 데이터 분석 확인
print(x_train[0,5])
print(x_train[0,12])
print("평균 방수의 표준 편차 ", np.std(x_train[:,5]))
print("평균 방수의 최대 값 ", np.max(x_train[:,5]))
print("평균 방수의 최소 값 ", np.min(x_train[:,5]))
print("인구 밀도의 표준 편차 ", np.std(x_train[:,12]))
print("인구 밀도의 최대값 ", np.max(x_train[:,12]))
print("인구 밀도의 최소값 ", np.min(x_train[:,12]))


# In[21]:


# 결측값 있는지 체크 : na(not a variable), nan(not a number) = 파이썬에서 False로 인식함
print(sum(np.isnan(x_train[:,5])))  #x_train[:,12] == False
print(sum(np.isnan(x_train[:,12])))
print(np.isnan(np.nan)) #값이 nan이면 True 반환


# In[22]:


# 히스토그램 출력 : 데이터 분포도 및 이상값 여부 체크
plt.hist(x_train[:,5]) #값별 개수
plt.title("[5]")
plt.show()
plt.hist(x_train[:,12])
plt.title("[12]")
plt.show()


# In[23]:


# 정규화 시킨 뒤 히스토그램
mean5 = np.mean(x_train[:,5]) #방 수의 평균
std5 = np.std(x_train[:,5]) #방 수의 표준편차
mean12 = np.mean(x_train[:,12])
std12 = np.std(x_train[:,12])
x_train[:,5] = (x_train[:,5] - mean5) / std5  #정규화
x_train[:,12] = (x_train[:,12] - mean12) / std12
# 히스토그램
plt.figure(figsize=(7,2)) #width, heigt
plt.subplot(1,2,1) # 1행 2열짜리 중 1번째
plt.hist(x_train[:,5])
plt.subplot(1,2,2) # 1행 2열짜리 중 2번째
plt.hist(x_train[:,12])
plt.show()

# test 데이터 정규화
x_test[:,5] = (x_test[:,5] - mean5) / std5  #정규화
x_test[:,12] = (x_test[:,12] - mean12) / std12


# In[24]:


# 필요 데이터만 분리
x_train = x_train[:,[5,12]]
x_test = x_test[:,[5,12]]
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[25]:


# 전처리 후 최종 데이터 값 확인
print(x_train[0])
print(x_test[0])
print(y_train[0])
print(y_test[0])


# In[26]:


#모델 구성
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
model = Sequential()  #모델 구성
model.add(Input(2,))  #input layer 구성(다중회귀)
model.add(Dense(1))  #Dense layer 구성(출력 값은 1개)
model.compile(loss="MSE", optimizer="SGD")  #compile 속성 지정


# In[27]:


fhist = model.fit(x_train, y_train, epochs=15) #정규화된 데이터, 결과 값, 훈련회수)


# In[28]:


#plot 그래프 그려보기
print(fhist.history.keys())
plt.plot(fhist.history["loss"])
#그래프상 15회 이상 훈련이 무의미함을 확인 : 위에서 100회를 15회로 수정


# In[30]:


#테스트 데이터로 예측 값 뽑아보기
y_pred = model.predict(x_test)
print(y_pred.shape)  #작성한 모델로 뽑은 예측 데이터
y_test = y_test.reshape(len(y_test), -1) #예측 데이터와 모양 맞추기
print(y_test.shape)  #정답 데이터


# In[32]:


#전체 예측값에 대한 정확율 추출
#정답값과 예측값의 각 차이를 절대 값 배열로 받기
#정닶값으로 나눈 뒤 1에서 마이너스 해주면 정확율 배열
y_acc = 1 - (np.abs(y_test - y_pred) / y_test)
y_avg = np.mean(y_acc)*100  #정확율 배열의 평균 : 전체 평균
print(f"평균 정확률은 {y_avg:.2f} %")

