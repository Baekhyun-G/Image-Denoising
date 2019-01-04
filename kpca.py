# import tensorflow as tf
# from tensorflow import keras
# # from PIL import Image
import numpy as np
import gzip
import matplotlib.pyplot as plt
import math
# from sklearn.model_selection import train_test_split
from scipy.spatial import distance
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images,784)
        return data
def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels
def func(a,l,c):
    return 1.+math.expm1(-(a*1.)/(2.*l*c))

y_train = extract_labels('train-labels-idx1-ubyte.gz',500)
x_train = extract_data('train-images-idx3-ubyte.gz', 500)
y_filter = np.where(y_train == 2)
x_train=x_train[y_filter]
l=len(np.array(y_filter).flatten())
# print(l)
# x_train =np.reshape(x_train, 1,28,28)

# print(x_train[0])
# print(y_train.shape)
# print(l)
cov=np.array(distance.cdist(x_train[:-1],x_train[:-1],'sqeuclidean'))
tot=np.square(l-1)
cov=cov.flatten()
vfunc=np.vectorize(func)
cov=vfunc(cov,l-1,0.001).reshape(l-1,l-1)
onen=np.full((l-1, l-1), 1./l-1)
cov=cov-2*np.matmul(onen,cov)+np.matmul(np.matmul(onen,cov),onen)

u,s,v=np.linalg.svd(cov, full_matrices=False)
u=np.array([u[i]*1./math.sqrt(s[i]*(l-1)) for i in range(l-1)])
u=u[:,:70]
alpha=np.matmul(arr1,u)
ans=np.matmul(u,np.transpose(alpha))

plt.figure(figsize=[5,5])


plt.subplot(131)
curr_img = np.reshape(x_train[l-1], (28,28))
plt.imshow(curr_img, cmap='gray')

plt.subplot(132)
curr_img = np.reshape(arr1, (28,28))
plt.imshow(curr_img, cmap='gray')


plt.subplot(133)
curr_img = np.reshape(ans, (28,28))
plt.imshow(curr_img, cmap='gray')
plt.show()
