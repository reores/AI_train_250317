#!/usr/bin/env python
# coding: utf-8

# In[18]:


#컨볼루션 레이어 : 합성곱층, 색상 값이 있는 모든 파라미터 값을 더한다. 특성층을 결정한다.
#특성맵 추출과정 : 컨볼루션 합성곱층과 플링층을 거쳐 특성을 추출하는 방식
#특성맵으로 추출된 특성을 완전연결층(flatten - Dense)에서 특성을 훈련한다.

# tf.keras.layers.Conv2D(
#     filters,         => int, the dimension of the output space 특성 차원맵 설정, 지정한 사이즈만큼 특성맵이 정의됨
#     kernel_size,     => int or tuple/list of 2 integer, specifying the size of the convolution window 커널사이즈 정수값 1개 주면 N * N 형태
#     strides=(1, 1),  => int or tuple/list of 2 integer, specifying the stride length of the convolution 커널사이즈 이동 step 1개주면 동일하게 이동
#     padding='valid', => string, either "valid" or "same" (case-insensitive). "valid" means no padding. 
#                         "same" results in padding evenly to the left/right or up/down of the input.
#                         When padding="same" and strides=1, the output has the same size as the input. 
#     activation=None
# )

# tf.keras.layers.GlobalAveragePooling2D => 단일 특성맵 전체의 평균을 산출한다.(완전연결층을 보완)
# tf.keras.layers.AveragePooling2D => 풀링 사이즈만큼 특성맵을 추출한다.
# tf.keras.layers.MaxPool2D(
#     pool_size=(2, 2), => int or tuple of 2 integers, factors by which to downscale (dim1, dim2).
#     strides=None,     => int or tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.
#     padding='valid'   => string, either "valid" or "same" (case-insensitive). "valid" means no padding.
#                          "same" results in padding evenly to the left/right or up/down of the input such that
#                           output has the same height/width dimension as the input. 
# )


# In[19]:





# In[20]:


import numpy as np
import matplotlib.pyplot as plt
img_res = tf.keras.utils.load_img(
    "./examp.jpg",    
    target_size=(52, 52),
    interpolation='nearest',
    keep_aspect_ratio=True
)
print(img_res) #객체 PIL : 파이선 기본 이미지 라이브러리
img_res = np.array(img_res)
print(img_res.shape)

plt.figure(figsize=(5,5))
plt.imshow(img_res)
plt.xticks([]); plt.yticks([]);
plt.show()


# In[21]:


# 데이터 정규화
img_res = img_res / 255.
print(img_res[14])


# In[51]:


#(52,52,3) -> (1,52,52,3)
img_res = img_res.reshape(1,52,52,3)
print(img_res.shape)

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D
test_model = Sequential()
test_model.add(Input((52, 52, 3))) #이미지 사이즈 3채널
test_model.add(Conv2D(5, 7, padding="same", activation="relu"))
#filter, kernel_size, strides
#특성(feature)맵 개수, 커널사이즈(특성을 뽑을 원본 이미지의 범위), 스트라이드 기본값(생략), 패딩 동일하게(생략시 이미지 사이즈 줄어듬)
#커널사이즈는 일반적으로 정의해야 한다. 5정도, 너무 작거나 크면 일반적인 특성을 뽑을 수 없기 때문에. 홀수로 선언하는게 대부분
#Q. 커널사이즈에 맞게 합성곱이라는 계산식이 정해져 있는데, 필터개수를 지정해도 똑같은 결과 값이 나오는건 아닌지???
#Q. 3채널이어서라면 필터(특성맵) 개수도 3개여야 하는건 아닌지?
#A. 정의한 필터 개수만큼 임의의 필터 N개를 만들어? 생성해? 합성곱을 진행한다.
#A. 컨볼루션 레이어에서는 필터 값이 랜덤하게? 생성된다?
#과대적합 : 커널사이즈나 풀사이즈를 늘린다.
#과소적합 : 커널사이즈나 풀사이즈를 낮춘다.
res = test_model(img_res)
print(res.shape) #특성맵

plt.imshow(res[0,:,:,0])
plt.show()
plt.imshow(res[0,:,:,1])
plt.show()
plt.imshow(res[0,:,:,4])
plt.show()


# In[57]:


#pooling : 특성맵에서 각 특징을 추출
#pool_size : 지정 범위 내에서 Max값을 추려내므로 값이 클수록 이미지가 뭉게진다.
# └ strides 값이 기본이라고 가정할때 pool_size가 클수록 이미지 사이즈가 줄어든다.
#strides : 특성맵에서 pool_size를 건너뛸 step같은 개념? 따라서 값이 클수록 이미지 사이즈가 줄어든다.
test_model = MaxPool2D(3, 2) #pool_size, strides, padding
res1 = test_model(img_res)
test_model = MaxPool2D(5, 3)
res2 = test_model(img_res)
print("res1: ", res1.shape, " res2: ", res2.shape) #특성맵
plt.figure(figsize=(5,5))
#res1
plt.subplot(2,3,1)
plt.imshow(res1[0,:,:,0])
plt.subplot(2,3,2)
plt.imshow(res1[0,:,:,1])
plt.subplot(2,3,3)
plt.imshow(res1[0,:,:,2])

#res2
plt.subplot(2,3,4)
plt.imshow(res2[0,:,:,0])
plt.subplot(2,3,5)
plt.imshow(res2[0,:,:,1])
plt.subplot(2,3,6)
plt.imshow(res2[0,:,:,2])

#AveragePooling2D
test_avg = AveragePooling2D(5, 3)
res3 = test_avg(img_res)
#res2
plt.subplot(1,3,1)
plt.imshow(res3[0,:,:,0])
plt.subplot(1,3,2)
plt.imshow(res3[0,:,:,1])
plt.subplot(1,3,3)
plt.imshow(res3[0])

plt.show()

