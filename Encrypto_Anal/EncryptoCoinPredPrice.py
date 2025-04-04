#!/usr/bin/env python
# coding: utf-8

# In[15]:


# 딥러닝 암호화폐 가격 예측
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utilpy import getCandleData, createX, integeration_xdata


# In[16]:


# 정의한 util에서 데이터 받아오기
candle_data = getCandleData("days", cname="BTC") #ttime의 기본 값은 days
x_datasets, y_datasets = createX(candle_data, 10) #전처리할 데이터와 문제파일의 추출 개수
x_datasets, featurelist = integeration_xdata(x_datasets)

print(x_datasets.shape)
print(y_datasets.shape)


# In[17]:


print(x_datasets[0])
print(featurelist)


# In[18]:


# 스캐터 그려서 산점도 보기
# opening_price = np.mean(x_datasets[:, :, 0], axis=1) #axis = 1 : 5개의 평균
# high_price = np.mean(x_datasets[:, :, 1], axis=1) 
# low_price = np.mean(x_datasets[:, :, 2], axis=1) 
# trade_price = np.mean(x_datasets[:, :, 3], axis=1) 
# candle_acc_trade_price = np.mean(x_datasets[:, :, 4], axis=1) 
# candle_acc_trade_volume = np.mean(x_datasets[:, :, 5], axis=1) 

# plt.figure(figsize=(8,3))
# plt.subplot(2,3,1)
# plt.scatter(opening_price, y_datasets[:,0], s=3)
# plt.subplot(2,3,2)
# plt.scatter(high_price, y_datasets[:,1], s=3)
# plt.subplot(2,3,3)
# plt.scatter(low_price, y_datasets[:,2], s=3)
# plt.subplot(2,3,4)
# plt.scatter(trade_price, y_datasets[:,3], s=3)
# plt.show()


# In[19]:


# 산점도 확인 후 연관성 없는 데이터 삭제
#axis = -1 맨 마지막(가장 안쪽)에서 [-2, -1] 뒤 2개 인덱스 삭제
x_datasets = np.delete(x_datasets, [-2, -1], axis=-1)
print(len(x_datasets))
print(x_datasets[0])


# In[20]:


# 데이터 정규화
# 주의! 특징별로(열별로) 정규화가 이루어지어야 함.
import sklearn
# scaler = sklearn.preprocessing.StandardScaler() : 2차원 이하만 작동한다.
# z = (x - u) / s
m1 = x_datasets[:,:,0].mean()
s1 = x_datasets[:,:,0].std()
m2 = x_datasets[:,:,1].mean()
s2 = x_datasets[:,:,1].std()
m3 = x_datasets[:,:,2].mean()
s3 = x_datasets[:,:,2].std()
m4 = x_datasets[:,:,3].mean()
s4 = x_datasets[:,:,3].std()

x_datasets[:,:,0] = (x_datasets[:,:,0] - m1) / s1
x_datasets[:,:,1] = (x_datasets[:,:,1] - m2) / s2
x_datasets[:,:,2] = (x_datasets[:,:,2] - m3) / s3
x_datasets[:,:,3] = (x_datasets[:,:,3] - m4) / s4

# 정답 데이터에 대한 정규화
# ymean = y_datasets.mean()
# ystd = y_datasets.std()
# y_datasets = (y_datasets - ymean) / ystd #복구화는 계산식 반대로


# In[21]:


print(x_datasets[-1])
#가장 안 차원의 같은 특성별 평균 값을 구한다. 시가, 종가, 최고가, 최저가
x_datasets = np.mean(x_datasets, axis=-1)
y_datasets = np.mean(y_datasets, axis=-1)
print(x_datasets.shape, y_datasets.shape)


# In[22]:


# 모델 구성
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
#과소적합으로 모델 복잡도 상향, 드랍아웃 제거 처리, 아담 학습률 속성 상향 제어하여 직접 선언
layer_adam = tf.keras.optimizers.Adam(0.005)
model.add(Input((x_datasets.shape[1],))) 
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="linear")) 
model.compile(loss="mae", optimizer=layer_adam, metrics=["mse"]) #l : mae, msd / o : adam, sgd


# In[23]:


# 모델 훈련
fhist = model.fit(x_datasets, y_datasets, epochs=500, batch_size=20)


# In[24]:


# 정답 수치가 크기 때문에 loss 값도 크다 : loss = 정답파일과의 오차 값
y_pred = model.predict(x_datasets)
print(y_pred.shape, y_datasets.shape)
# (194, 5, 4) 데이터로 모델 훈련 => (194, 5, 1) 모양의 예측결과가 반환됨
y_pred = y_pred.reshape(y_pred.shape[0])
print(y_pred.shape)
print(y_pred[0])


# In[25]:


plt.plot(y_datasets, y_datasets, color="red")
plt.scatter(y_datasets, y_pred, s=3)
plt.show()


# In[26]:


# 가격정보 예측
print("오늘의 가격 정보 예측: ", end="")
print(f"최저: {y_pred[-2]*0.9:.2f}, 최고: {y_pred[-2]*1.1:.2f}, 평균: {y_pred[-2]:.2f}")
print("내일의 가격 정보 예측: ", end="")
print(f"최저: {y_pred[-1]*0.9:.2f}, 최고: {y_pred[-1]*1.1:.2f}, 평균: {y_pred[-1]:.2f}")
print("내일의 가격 변동률 예측: ", end="")
print(f"최저: {((y_pred[-1])/(y_pred[-2])-1)*100*0.9:.2f}%,\
        최고: {((y_pred[-1])/(y_pred[-2])-1)*100*1.1:.2f}%,\
        평균: {((y_pred[-1])/(y_pred[-2])-1)*100:.2f}%")


# In[27]:


# 2025-04-03 일간예측 : 5일 기준
# 오늘의 가격 정보 예측: 최저: 111620865.60, 최고: 136425502.40, 평균: 124023184.00
# 내일의 가격 정보 예측: 최저: 112885005.60, 최고: 137970562.40, 평균: 125427784.00
# 내일의 가격 변동률 예측: 최저: 1.02%,        최고: 1.25%,        평균: 1.13%

# 2025-04-03 일간예측 : 10일 기준
# 오늘의 가격 정보 예측: 최저: 112476772.80, 최고: 137471611.20, 평균: 124974192.00
# 내일의 가격 정보 예측: 최저: 113607691.20, 최고: 138853844.80, 평균: 126230768.00
# 내일의 가격 변동률 예측: 최저: 0.90%,        최고: 1.11%,        평균: 1.01%

# 2025-04-03 주간예측 : 10주 기준
# 금주의 가격 정보 예측: 최저: 107571902.40, 최고: 131476769.60, 평균: 119524336.00
# 차주의 가격 정보 예측: 최저: 111640564.80, 최고: 136449579.20, 평균: 124045072.00
# 차주의 가격 변동률 예측: 최저: 3.40%,        최고: 4.16%,        평균: 3.78%

# 2025-04-03 월간예측 : 10개월 기준
# 이번달의 가격 정보 예측: 최저: 143542598.40, 최고: 175440953.60, 평균: 159491776.00
# 다음달의 가격 정보 예측: 최저: 115999984.80, 최고: 141777759.20, 평균: 128888872.00
# 다음달의 가격 변동률 예측: 최저: -17.27%,        최고: -21.11%,        평균: -19.19%

# 2025-04-03 시간예측 : 1시간 기준
# 현재의 가격 정보 예측: 최저: 108329544.00, 최고: 132402776.00, 평균: 120366160.00
# 1시간 뒤 가격 정보 예측: 최저: 104712472.80, 최고: 127981911.20, 평균: 116347192.00
# 1시간 뒤 가격 변동률 예측: 최저: -3.01%,        최고: -3.67%,        평균: -3.34%


# In[28]:


# 그래프로 그려보기
plt.plot(y_datasets, label="True")
plt.plot(y_pred, label="Pred")
plt.legend()
plt.show()

