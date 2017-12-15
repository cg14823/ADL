import numpy as np
import scipy.ndimage
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cPickle as pickle
import time
import random
import scipy.ndimage
from skimage import exposure

def whitening(dataset):
    x = 0
    for i in range(dataset["X_train"].shape[0]):
        if (x % 100 == 0):
            print("{} of {}".format(x, dataset["X_train"].shape[0]))
        for j in range(3):
            mean = np.mean(dataset["X_train"][i][:,:,j])
            std = np.std(dataset["X_train"][i][:,:,j])
            dataset["X_train"][i][:,:,j] = (dataset["X_train"][i][:,:,j]-mean)/std
        x += 1
    return dataset

def batch_generator(dataset, group, batch_size=1):

	idx = 0
	dataset = dataset[0] if group == 'train' else dataset[1]

	dataset_size = len(dataset)
	indices = range(dataset_size)
	np.random.shuffle(indices)
	while idx < dataset_size:
		chunk = slice(idx, idx+batch_size)
		chunk = indices[chunk]
		chunk = sorted(chunk)
		idx = idx + batch_size
		yield [dataset[i][0] for i in chunk], [dataset[i][1] for i in chunk]

def applyMotionBlur(imageIn):
    zA = [0,0,0]
    oA = [1,1,1]
    tA = [2,2,2]
    motionBlurKernel = np.array([[zA, zA, zA, zA, zA],[zA, zA, zA, zA, zA],[oA, oA, tA, oA, oA],[zA, zA, zA, zA, zA],[zA, zA, zA, zA, zA]])
    motionBlurKernel = np.divide(motionBlurKernel,18.)
    imageOut = scipy.ndimage.filters.convolve(imageIn,motionBlurKernel,mode='nearest',cval=0.0)
    imageOut = np.clip(imageOut,0,1)
    return imageOut

def rgb2yuv(img):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
    yuv = np.dot(img,m)
    yuv[:,:,1:]+=128.0
    return yuv

def YUV2RGB( yuv ):
       
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
     
    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304
    return rgb

def main():
    data_set = pickle.load(open('dataset.pkl','rb'))
    generate_motherfucker =batch_generator(data_set, 'train')
    (x, y) = generate_motherfucker.next()
    #print(x)
    print(np.shape(x))
    print(np.shape(x[0]))
    #print(y)
    print(np.shape(y))
    fig = plt.figure()
    plt.imshow(x[0])
    plt.show()
    fig.savefig('NoBlur.png')
    x2 = rgb2yuv(x[0])
    x2 = x2[:,:,0]
    x2 = (x2 / 255.).astype(np.float32)
    x2 = (exposure.equalize_adapthist(x2,) - 0.5)
    #x2 = x2.reshape(x2.shape + (1,))
    print(np.shape(x2))
    fig = plt.figure()
    plt.imshow(x2,cmap='gray')
    plt.show()
    fig.savefig('Blur.png')
    print(x[0])
    print(x2)


main()