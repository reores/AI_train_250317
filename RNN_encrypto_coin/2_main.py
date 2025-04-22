#!/usr/bin/env python
# coding: utf-8

#정수형 데이터를 표준편차로 변경
#임베딩 : 토큰화된 문자를 정수데이터로 변경이므로 생략가능?
#import sklearn
#scaler = sklearn.preprocessing.StandardScaler()
#fit : 평균과 표준편차 계산
#transform : 중심을 잡고 확장하여 표준화를 수행
#fit_tranform : 동시 수행

"""
    가상화폐 일간 가격 분석
"""

import tensorflow as tf
import numpy as np
import get_data
import matplotlib.pyplot as plt
from get_data import get_conindata, create_rnn_data, get_xpred
import pickle

#표준편차로 정규화 z = (x - u) / s
#1. 스케일 조정 함수
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
    with open("BTC_days.pic", "wb") as fp:
        pickle.dump({"x_scaler_mean":x_scaler_mean, "x_scaler_std":x_scaler_std}, fp)
    return x_data

#2. 예측할 데이터 스케일 적용함수
def pred_scaler(x_data):
    with open("BTC_days.pic", "rb") as fp:
        scdata = pickle.load(fp)
        x_scaler_mean = scdata["x_scaler_mean"]
        x_scaler_std = scdata["x_scaler_std"]    
    x_data = x_data.astype(np.float64)
    for i in range(len(x_data[0][0])):        
        x_data[:,:,i] = (x_data[:,:,i] - x_scaler_mean[i]) / x_scaler_std[i] #각 필드별 정규분호        
    return x_data    

#3. 데이터 수신 및 생성
rawdata = get_conindata("BTC", to="2016-03-02 00:00:00")    
x_data, y_data = create_rnn_data(rawdata)
print(x_data.shape, y_data.shape)
x_data = scalerX(x_data)
print(x_data[0][0])

#4. ConvLSTM1D 모델구성
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, ConvLSTM1D, Bidirectional, Reshape, GlobalMaxPool1D, Dropout, GlobalAveragePooling1D, AveragePooling1D, MaxPooling1D, Flatten

#ConvLSTM1D 은 4차원 텐서로 전달되어야 함
#x_data = np.expand_dims(x_data, axis=-1)
#model.add(Input((30, 5, 1)))

#컨볼루션 : 성능 우수
#LSTM : 과거데이터 영향력 향상
model = Sequential()
model.add(Input((30, 5)))
model.add(Reshape((30,5,1)))
conv_lstm1 = ConvLSTM1D(
    filters=16, #N*N size 필터(특성)
    kernel_size=3, #특성을 뽑을 범위, 작을수록 세밀하게
    strides=1,
    padding='same',
    dropout=0.3,
    recurrent_dropout=0.5,
    return_sequences=True #전체 시퀀스 반환
)
conv_lstm2 = ConvLSTM1D(
    filters=32, # 특성값은 점점 확장
    kernel_size=5,
    strides=2,
    padding='same',
    dropout=0.4,    
    recurrent_dropout=0.4,
    return_sequences=False #마지막 시퀀스 반환
)

model.add(Bidirectional(conv_lstm1)) # 양방향 모델로 구성
model.add(Bidirectional(conv_lstm2))
#model.add(conv_lstm1) #단방향 모데로 구성
#model.add(conv_lstm2)
model.add(AveragePooling1D())
#model.add(Dense(256, activation="relu"))
model.add(Flatten())
model.add(Dropout(0.4, seed=123))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.4, seed=123))
model.add(Dense(1, activation="linear")) #선형회귀 모델 : 예상값 1개만 도출하므로
adam = tf.keras.optimizers.Adam(learning_rate=0.00009)
model.compile(loss="mse", optimizer=adam, metrics=["mae"])

print(x_data.shape, y_data.shape)

#5. 모델 훈련 및 체크포인트 콜백 함수 적용
filepath = "{epoch:02d}-{val_loss:.2f}.keras"
mck = tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_loss',    
    save_best_only=True,    
    mode='auto',
    save_freq='epoch'
)

y_data = y_data/100000000. #정답 값이 너무 클 경우 훈련이 비효율적
fhist = model.fit(x_data, y_data, validation_split=0.2, epochs=15, callbacks=[mck], batch_size=len(x_data)//10)
# fhist = model.fit(x_data, y_data, validation_split=0.2, epochs=15, batch_size=len(x_data)//10)

#6. 모델 훈련결과 시각화
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

#7. 최적화된 모델 호출
model = tf.keras.models.load_model("BTC_days_T30.keras")

#8. 산점도 그래프 및 선형 그래프로 정답과 예측값 일치성 시각화
y_pred = model.predict(x_data)
plt.scatter(y_data, y_pred, label="pred _ acc", s = 1, color="red")
plt.plot(y_data, y_data, label="true _ acc")
plt.legend()
plt.show()

#9. 예측할 오늘의 데이터와 어제의 데이터 수신
#내일의 가격 예측 BTC_days_T30
#최근 30일 데이터 1묶음을 수신해와 예측모델에 적용
model = tf.keras.models.load_model("BTC_days_T30.keras")
x_today, x_yesday, y_cur_price = get_xpred(coinname="BTC", getunit="days", timm="", timestep=30)
x_today = pred_scaler(np.array([x_today]))
x_yesday = pred_scaler(np.array([x_yesday]))
print(x_today.shape)
print(x_yesday.shape)

#10. 어제 데이터의 정확도 및 예측값의 오차율 출력과 오늘 데이터의 예측값 출력
y_yes_pred = model.predict([x_yesday])
y_tod_pred = model.predict([x_today])
print("현재가격: ", y_cur_price, "현재 예측가격: ", y_tod_pred[0][0]*100000000, "오차율: ", (1-int(abs(y_tod_pred[0][0]*100000000/y_cur_price)*100)/100)*100, "%")
print("내일의 예측 가격은: ",y_tod_pred[0][0]*100000000)

