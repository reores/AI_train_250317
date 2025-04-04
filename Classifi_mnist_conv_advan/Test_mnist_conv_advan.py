#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 만들어진 Convolution 모델을 사용해 실제이미지 분류해보기
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


SPATH = r"checkpt/"
opt_model = tf.keras.models.load_model(f"{SPATH}select_6-0.05.keras")


# In[3]:


# tf.keras.utils.load_img 이미지 불러오기
img3 = np.array(tf.keras.utils.load_img(r"test_img\3.png", color_mode='grayscale'))
img4 = np.array(tf.keras.utils.load_img(r"test_img\4.png", color_mode='grayscale'))
img6 = np.array(tf.keras.utils.load_img(r"test_img\6.png", color_mode='grayscale'))
img7 = np.array(tf.keras.utils.load_img(r"test_img\7.png", color_mode='grayscale'))
img8 = np.array(tf.keras.utils.load_img(r"test_img\8.png", color_mode='grayscale'))
img9 = np.array(tf.keras.utils.load_img(r"test_img\9.png", color_mode='grayscale'))
test_img = np.array([img3, img4, img6, img7, img8, img9])
print(test_img.shape)
test_img = 255-test_img.reshape(6, 28, 28, 1)
print(test_img.shape)
plt.imshow(test_img[0], cmap="gray")
plt.show()                          


# In[4]:


y_pred = opt_model.predict(test_img)
print(y_pred.shape)


# In[6]:


plt.figure(figsize=(8,3))
for i, d in enumerate(test_img):
    plt.subplot(2, 3, i+1)
    plt.imshow(d, cmap="gray")
    plt.title(np.argmax(y_pred[i]))
    plt.xticks([]); plt.yticks([]);
plt.show()

