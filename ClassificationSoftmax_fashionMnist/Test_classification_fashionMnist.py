#!/usr/bin/env python
# coding: utf-8

# In[64]:


#저장된 모델 불러오기 : 옷 분류 모델
import tensorflow as tf
model = tf.keras.models.load_model(r"./save_model/fashion_mnist_classification.keras")
model.summary() #모델 요약(기본정보) : 파라미터 개수 등


# In[65]:


#저장된 encoder 객체 불러오기 rb : read binary
# r"" : 역슬러시 그대로 적용
import pickle
encoder = None
with open(r"./save_model/fashion_mnist.classification_encoder", "rb") as fp:
    encoder = pickle.load(fp)
encoder.one_hot_to_label([[0,1,0,0,0,0,0,0,0,0]])


# In[66]:


#이미지 불러오기
# 이미지 1개 불러오기
# tf.keras.utils.load_img(
#     path,
#     color_mode='rgb',
#     target_size=None,
#     interpolation='nearest',
#     keep_aspect_ratio=False
# )

#디렉토리 기준 이미지 여러개 가져오기
#경로 하위에 있는 모든 이미지 파일을 불러온다.
#상이 폴더를 구분지었던 이유는 실제 정답과 예측값을 체크하기 위해
datasets = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:\jupyter_work\data\test_img",
    labels=None,    
    color_mode='grayscale',    
    image_size=(28, 28), #모델 훈련사이즈에 맞게?
    interpolation='nearest'
)
print(type(datasets))
print("========================")
tmp = datasets.as_numpy_iterator()
for d in tmp:
    print(d.shape)
    dr = d.reshape(len(d), len(d[0][0]), len(d[0][1]))
    print(dr.shape)
print("========================")

import numpy as np
x_real = np.array([ d for d in datasets ][0])
print(x_real.shape)
x_real = x_real.reshape(len(x_real), 28, 28)
# 배경 없애기 위해 rgb 값을 인버스
x_real = 255.-x_real
print(x_real.shape)
print(x_real[0][1])  #가장 윗줄
print(x_real[0][-1]) #가장 아랫줄
#이미지 전처리 : 최상단, 최하단 줄에서 max 값을 판단해 그 이하 값은 모두 0처리
for data in x_real:
    backcolor_value = max(data[0]) if max(data[0]) - max(data[-1]) > 0 else max(data[-1])
    maskdata = data > backcolor_value # loop 값이 배경값보다 크면 true : 배경이 아닌 실제 이미지 데이터
    data = data * maskdata #false = 0 이므로, 곱셈으로 마스킹 처리 : 불필요 배경 이미지 데이터를 0으로 변경
    print(data[0])


# In[67]:


import matplotlib.pyplot as plt
y_real = []
label_list = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
for ix in range(len(x_real)):
    plt.figure(figsize=(1,1))
    plt.imshow(x_real[ix])
    plt.xticks([]); plt.yticks([]); 
    plt.show()
    for i in range(len(label_list)):
        print(f"{i+1}. {label_list[i]}\t",end="")
    print()
    usersel = input("이미지 정답 라벨을 번호를 입력하세요\n")    
    y_real.append(label_list[int(usersel) - 1])
print(y_real)


# In[68]:


print(x_real.shape)
print(y_real)
#예측 실행
y_pred_real = model.predict(x_real)
print(y_pred_real.shape)


# In[70]:


print(y_pred_real) #원핫인코딩 된 값으로 리턴됨
y_pred_label = encoder.one_hot_to_label(y_pred_real) #예측 라벨값을 담아둘 별도의 변수
print(y_pred_label)
plt.figure(figsize=(7,7))
for i in range(len(x_real)):
    plt.subplot(1,4,i+1) #3번째 파라미터가 그래프(여기서는 이미지)의 순서, 1부터 시작
    plt.imshow(x_real[i], cmap="gray")
    clr = "blue" if y_real[i] == y_pred_label[i] else "red"
    plt.title("True : "+ y_real[i], color=clr)
    plt.xlabel("Pred : "+ y_pred_label[i])
plt.show()

#250324 추가
for ix in range(len(y_pred_label)):
    plt.subplot(1,4,ix+1)
    plt.imshow(x_real[ix])
    clr = "blue"
    if y_real[ix] != y_pred_label[ix]:clr="red"    
    plt.xticks([]); plt.yticks([]);
    plt.title(y_real[ix], color=clr)
    plt.xlabel(y_pred_label[ix], color=clr)
plt.show()


# In[ ]:




