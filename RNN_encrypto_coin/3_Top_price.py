#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import get_data
import matplotlib.pyplot as plt
from get_data import get_conindata, get_xpred, high_create_rnn_data, get_high_xpred
import pickle
pre_fix = "high_"

#표준편차로 정규화 z = (x - u) / s
def scalerX(x_data):
    x_scaler_mean = []
    x_scaler_std = []
    # 각 항목별로 정규화 처리
    x_data = x_data.astype(np.float64)
    for i in range(len(x_data[0][0])):
        xm = x_data[:,:,i].mean()
        xs = x_data[:,:,i].std()
        x_data[:,:,i] = (x_data[:,:,i] - xm) / xs #각 필드별 정규분호
        x_scaler_mean.append(xm)
        x_scaler_std.append(xs)
    with open(pre_fix+"BTC_days.pic", "wb") as fp:
        pickle.dump({pre_fix+"x_scaler_mean":x_scaler_mean, pre_fix+"x_scaler_std":x_scaler_std}, fp)
    return x_data

def pred_scaler(x_data):
    with open(pre_fix+"BTC_days.pic", "rb") as fp:
        scdata = pickle.load(fp)
        x_scaler_mean = scdata[pre_fix+"x_scaler_mean"]
        x_scaler_std = scdata[pre_fix+"x_scaler_std"]    
    x_data = x_data.astype(np.float64)
    for i in range(len(x_data[0][0])):        
        x_data[:,:,i] = (x_data[:,:,i] - x_scaler_mean[i]) / x_scaler_std[i] #각 필드별 정규분호        
    return x_data    

#데이터 수신 및 생성
rawdata = get_conindata("BTC", to="2016-03-02 00:00:00")    
x_data, y_data = high_create_rnn_data(rawdata)
print(x_data.shape, y_data.shape)
x_data = scalerX(x_data)
print(x_data[0][0])

# 모델구성
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, LSTM, ConvLSTM1D, Bidirectional, Reshape, GlobalMaxPool1D, Dropout, GlobalAveragePooling1D, AveragePooling1D, MaxPooling1D, Flatten
tf.random.set_seed(123)
np.random.seed(123)

model = Sequential()
model.add(Input((30, 1))) #30일, 1개 데이터(최고가)

ls1 = LSTM(
    units=32, #출력차원
    dropout=0.3,
    recurrent_dropout=0.3,    
    return_sequences=True #True : 전체 시퀀스 반환
)
model.add(ls1)
#res = model(x_data)
#print(res.shape) # units가 8이었을 때 : (3370, 30, 1) → (3370, 30, 8)

ls2 = LSTM(
    units=32, 
    dropout=0.3,
    recurrent_dropout=0.3,
    return_sequences=True 
)
model.add(ls2)
# res = model(x_data)
# print(res.shape) #  units가 16이었을 때 : (3370, 30, 1) → (3370, 30, 8) → (3370, 16)

ls3 = LSTM(
    units=64, #16차원
    dropout=0.2,
    recurrent_dropout=0.3,    
    return_sequences=False #False : 마지막 출력 반환
)
model.add(ls3)
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4, seed=123))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="linear")) #선형회귀 모델 : 예상값 1개만 도출하므로
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss="mse", optimizer=adam, metrics=["mae"])

# 훈련 전 데이터 구조 확인
print(x_data.shape, y_data.shape)

#모델 훈련 및 저장(체크포인트)
filepath = pre_fix+"{epoch:02d}-{val_loss:.2f}.keras"
mck = tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_loss',    
    save_best_only=True,    
    mode='auto',
    save_freq='epoch'
)
y_data = y_data / 100000000.
fhist = model.fit(x_data, y_data, validation_split=0.2, epochs=50, callbacks=[mck], batch_size=len(x_data)//10)

plt.subplot(1,2,1)
plt.plot(fhist.history["loss"], label="train_mse")
plt.plot(fhist.history["val_loss"], label="valid_mse")
plt.legend()
plt.title("MSE")
plt.subplot(1,2,2)
plt.plot(fhist.history["mae"], label="train_mae")
plt.plot(fhist.history["val_mae"], label="valid_mae")
plt.legend()
plt.title("MAE")
plt.show()

#최적화된 모델 호출
model = tf.keras.models.load_model(pre_fix+"BTC_days.keras")
#예측하기
y_pred = model.predict(x_data)
plt.scatter(y_data, y_pred, label="pred _ acc", s = 1, color="red")
plt.plot(y_data, y_data, label="true _ acc")
plt.legend()
plt.show()

#내일의 가격 예측 BTC_days_T30
#최근 30일 데이터 1묶음을 수신해와 예측모델에 적용
model = tf.keras.models.load_model(pre_fix+"BTC_days.keras")
x_today, x_yesday, y_cur_price = get_high_xpred(coinname="BTC", getunit="days", timm="", timestep=30)
# x_today = x_today.reshape(1, 30, 1)
x_today = pred_scaler(np.array([x_today]))
x_yesday = pred_scaler(np.array([x_yesday]))
print(x_today.shape)
print(x_yesday.shape)

y_yes_pred = model.predict([x_yesday])
y_tod_pred = model.predict([x_today])
print("현재 최고가격: ", y_cur_price, "현재 예측 최고가격: ", y_tod_pred[0][0]*100000000, "오차율: ", (1-int(abs(y_tod_pred[0][0]*100000000/y_cur_price)*100)/100)*100, "%")
print("내일의 예측 최고 가격은: ",y_tod_pred[0][0]*100000000)

