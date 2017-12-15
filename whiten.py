import numpy as np
import scipy.ndimage
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    zA = [1,1,1]
    oA = [2,1,2]
    motionBlurKernel = np.array([[zA, zA, zA],[oA, oA, oA],[zA, zA, zA]])
    motionBlurKernel = np.divide(motionBlurKernel,27.)
    print(motionBlurKernel)
    imageOut = scipy.ndimage.filters.convolve(imageIn,motionBlurKernel,mode='nearest',cval=0.0)
    imageOut = np.clip(imageOut,0,1)
    return imageOut

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
    x2 = applyMotionBlur(x[0])
    fig = plt.figure()
    plt.imshow(x2)
    plt.show()
    fig.savefig('Blur.png')
    print(x[0])
    print(x2)


main()