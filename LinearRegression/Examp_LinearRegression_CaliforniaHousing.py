#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt

datasets = sklearn.datasets.fetch_california_housing()
print(datasets.keys())

#data 문제데이터, target 정답데이터, feature_names 문제데이터 특성, target_names 정답 특성


# In[2]:


# 1. 문제제이터 x_data와 정답데이터 y_data를 분리하시오. - 데이터 수집
x_data = datasets["data"]  #8개의 특성을 갖는 주택 데이터
y_data = datasets["target"]  #data의 결과 값(정답 값), 주택 평균 가격, 단위 10만달러
print(x_data.shape)
print(y_data.shape)
print(x_data[0])
print(y_data[0])

# 2. feature_names를 feature 변수로 분리하여 문제파일의 특성을 기재하시오. - 데이터 분석
feature = datasets["feature_names"]
print(feature)
# 0 MedInc 블록 그룹의 평균 소득
# 1 HouseAge 블록 그롭의 평균 집 년한
# 2 AveRooms 평균 객실 수
# 3 AveBedrms 평균 침실 수
# 4 Population 블록별 인구수
# 5 AveOccup 평균 가구 구성원 수
# 6 Latitude 위도
# 7 Longitude 경도


# In[3]:


# 3. 위도와 경도는 평균 값을 이용하고 모든 특성의 산점도 그래프를 그려 연관성을 시각화 하시오.
plt.figure(figsize=(12,8))
for ix in range(len(x_data[0])):
    plt.subplot(4, 2, ix+1)
    plt.scatter(x_data[:,ix], y_data, s=1)  #문제데이터(x_data)의 모든 행에 ix번째 데이터
    plt.title(feature[ix])
plt.show()

# 3-1. 위도와 경도 평균 값 이용
plt.scatter((x_data[:,6]+x_data[:,7]) / len(x_data), y_data, s=1)
plt.title("AVG(la+lo)")
plt.show()


# In[4]:


import pandas as pd
# 판다스로 데이터 요약
df = pd.DataFrame(x_data, columns=feature)
print(df.describe()) #데이터 요약 보기


# In[5]:


# 이상데이터 확인 : 2(방 수), 3(침실 수), 4(인구 수), 5(구성원 수)
# 2(40), 3(50), 4(8000), 5(200) : 데이터 임계치 산정
plt.figure(figsize=(8,4))
plt.subplot(2, 2, 1)
plt.hist(x_data[:,2])
plt.subplot(2, 2, 2)
plt.hist(x_data[:,3])
plt.subplot(2, 2, 3)
plt.hist(x_data[:,4])
plt.subplot(2, 2, 4)
plt.hist(x_data[:,5])
plt.show()


# In[6]:


import numpy as np
# 2(40), 3(5), 4(8000), 5(200) : 데이터 임계치 산정
# 임계치 산정하는 함수 선언
def cutData(xdata, ydata) : #이상치 데이터 커팅
    # 마스킹할 인덱스 argwhere로 추출    
    tar2 = np.argwhere(xdata[:,2] >= 40)
    # numpy delete 함수로 삭제할 인덱싱 마스킹하여 삭제 처리
    xdata = np.delete(xdata, tar2, axis=0)
    ydata = np.delete(ydata, tar2, axis=0)    
    
    tar3 = np.argwhere(xdata[:,3] >= 5)    
    xdata = np.delete(xdata, tar3, axis=0)    
    ydata = np.delete(ydata, tar3, axis=0)
    
    tar4 = np.argwhere(xdata[:,4] >= 8000)    
    xdata = np.delete(xdata, tar4, axis=0)
    ydata = np.delete(ydata, tar4, axis=0)
    
    tar5 = np.argwhere(xdata[:,5] >= 200)        
    xdata = np.delete(xdata, tar5, axis=0)
    ydata = np.delete(ydata, tar5, axis=0)
    
    print(xdata.shape)
    print(ydata.shape)
    
    return xdata, ydata

x_data, y_data = cutData(x_data, y_data)  #언패킹


# In[7]:


# 이상치, 결측치 데이터 제거 후 히스토그램 다시 출력
plt.figure(figsize=(8,4))
plt.subplot(2, 2, 1)
plt.hist(x_data[:,2])
plt.subplot(2, 2, 2)
plt.hist(x_data[:,3])
plt.subplot(2, 2, 3)
plt.hist(x_data[:,4])
plt.subplot(2, 2, 4)
plt.hist(x_data[:,5])
plt.show()


# In[8]:


# 데이터 분할(훈련용과 테스트용으로) : test_size=0.2 는 전체 데이터 중 80%훈련용, 20%를 테스트용으로 분할
x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.2, random_state=111)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[9]:


# 4. x_train, x_test를 정규화한 뒤 데이터를 확인
# 이때, x_train의 평균과 표준편차를 이용하시오.
for ix in range(len(x_train[0])):
    x_train[:,ix] = (x_train[:,ix] - np.mean(x_train[:,ix])) / np.std(x_train[:,ix])
    x_test[:,ix] = (x_test[:,ix] - np.mean(x_test[:,ix])) / np.std(x_test[:,ix])
print(x_train[0])
print(x_test[0])

# x_train에는 특성값 8개가 들어있으므로, 평균값과 표준편차, 정규화를 한꺼번에 처리하면 안된다. 위처럼 구성
# mean = np.mean(x_train)
# std = np.std(x_train)
# x_train = (x_train - mean) / std
# x_test = (x_test - mean) / std
# print(x_train[0])
# print(x_test[0])


# In[10]:


# 5. 순서모델을 구성하고 입력층과 출력층을 작성하시오
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Input((8,)))  #특성 값이 8개짜리이므로, 다중 선형모델로
model.add(Dense(1))

# 6. 모델을 컴파일 하세요(손실함수는 MSE, 최적화함수는 경사하강법을 사용하세요) : MAE(평균 절대 편차) / MSE(평균 제곱 오차)
model.compile(loss="MSE", optimizer="SGD")


# In[11]:


# 훈련시켜보기
fhist = model.fit(x_train, y_train, epochs = 300, batch_size=len(x_train)//5)


# In[12]:


# 7. 훈련결과 그래프 그려보기
plt.plot(fhist.history["loss"])
plt.show()


# In[13]:


# 테스트 데이터를 측정 후 실제 정답과 예측 정답의 정확률을 측정해보시오.
y_pred = model.predict(x_test)
print(y_pred.shape)
y_test = y_test.reshape(len(y_test), -1)
print(y_test.shape)

y_acc = 1 - (np.abs(y_test - y_pred) / y_test)  # 행렬연산 : 1 - ((오차 : 정답 값 - 예측 값) / 정답 값), 오차율이 아닌 정답률을 구하므로 1에서 차감
y_avg = np.mean(y_acc) * 100  # 백분율로 환산
print(f"정확률은 {y_avg:.2f} %입니다.")

