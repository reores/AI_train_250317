#!/usr/bin/env python
# coding: utf-8

# In[54]:


# 모델에서 Ealry Stopping, save point 실습
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# 시드 값 고정
import random
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)


# In[43]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(x_train[0][14], y_train[0])


# In[44]:


# 데이터 섞어주기
# sklearn.utils.shuffle(*arrays, random_state=None, n_samples=None)
# *arrays : 개수가 동일한 데이터 넣는 순서대로 리턴됨
# 사이킷런 셔플은 원핫인코딩 전에 수행해주기
x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=123)
x_test, y_test = sklearn.utils.shuffle(x_test, y_test, random_state=123)


# In[45]:


# 데이터 전처리
# min-max-scaler
x_train = x_train / 255.
x_test = x_test / 255.
# 정답 데이터 원핫인코딩
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)
print(x_train[0][14], y_test[0])


# In[46]:


# conv를 사용하기 위해 x 데이터를 줄 단위에서 픽셀 단위로 변경
# (60000, 28, 28) => (60000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape, x_test.shape)


# In[47]:


# 모델 체크포인트 정의
import os
SPATH = r"checkpt/"
if not os.path.exists(SPATH):
    os.mkdir(SPATH)
filepath = SPATH+"{epoch}-{val_loss:.2f}.keras"
mcp = tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

# 모델 조기종료 정의
espp = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc',    
    patience=10,
    verbose=1,
    restore_best_weights=True
)


# In[48]:


# 모델 구성
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout


# In[49]:


# 모델 구성
model = Sequential()
model.add(Input((x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(Conv2D(10, 3, padding="same", activation="relu")) #특성맵의 개수 10, 커널 사이즈 3
model.add(MaxPool2D(4, 1)) #pool_size 4, strides 1
model.add(Conv2D(40, 3, padding="same", activation="relu")) 
model.add(MaxPool2D(4, 1))
model.add(Flatten()) #완전층으로 연결
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation="softmax")) #최종 답 10개, softmax 다중분류로
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"]) # categorical_crossentropy 2개 이상


# In[50]:


# 모델 훈련
fhist = model.fit(x_train, y_train, validation_split=0.1, batch_size=30, epochs=200, callbacks=[mcp, espp])


# In[51]:


# Restoring model weights from the end of the best epoch: 6.
# 1800/1800 [==============================] - 81s 45ms/step - loss: 0.0270 - acc: 0.9920 - val_loss: 0.0642 - val_acc: 0.9887
# Epoch 16: early stopping
# 현재 모델은 6회차 훈련 가중치를 가지고 있다.


# In[52]:


# 모델 저장
model.save(f"{SPATH}val_acc99.keras")


# In[55]:


# 그래프 그려보기
plt.subplot(1,2,1)
plt.plot(fhist.history["loss"], label="train_loss")
plt.plot(fhist.history["val_loss"], label="valid_loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(fhist.history["acc"], label="train_acc")
plt.plot(fhist.history["val_acc"], label="valid_acc")
plt.legend()
plt.show()


# In[58]:


# 성능이 가장 좋았던 모델 불러오기 select_6-0.05.keras
optimal_model = tf.keras.models.load_model(f"{SPATH}select_6-0.05.keras")
res_opti = optimal_model.evaluate(x_test, y_test) # 저장된 모델 중 성능이 가장 좋은
res_old = model.evaluate(x_test, y_test) # 조기종료 모델
print(f"선택 모델의 손실도: {res_opti[0]}, 정확율 : {res_opti[1]}")
print(f"조기종료 모델의 손실도: {res_old[0]}, 정확율 : {res_old[1]}")


# In[70]:


rarr = np.random.randint(0, len(x_test), 10)
print(rarr)


# In[71]:


# 그래프 그리기
y_pred = optimal_model.predict(x_test)
plt.figure(figsize=(8,3))
for i, d in enumerate(rarr):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i], cmap="gray")
    clr = "red" if np.argmax(y_test[i]) != np.argmax(y_pred[i]) else "blue"
    plt.title(np.argmax(y_test[i]), color=clr)
    plt.xlabel(np.argmax(y_pred[i]))
plt.show()

