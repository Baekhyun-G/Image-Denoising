import numpy as np
import gzip
import matplotlib.pyplot as plt
import math
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

cov=np.cov(x_train[:-1],rowvar=False)
print(x_train[:-1].shape)
print(l)
arr1=x_train[l-1].flatten()
arr1=arr1+500.* np.random.normal(loc=0.0, scale=0.1**0.5, size=arr1.shape)
arr1=np.clip(arr1,0.,255.)

u,s,v=np.linalg.svd(cov, full_matrices=False)

u=u[:,:70]
alpha=np.matmul(arr1,u)
ans=np.matmul(u,np.transpose(alpha))
numer=distance.cdist(ans.reshape(1,784), x_train[l-1].reshape(1,784), 'euclidean')
numer1=distance.cdist(arr1.reshape(1,784), x_train[l-1].reshape(1,784), 'euclidean')

print(numer/numer1)


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
