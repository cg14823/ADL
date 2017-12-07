import numpy as np

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
	dataset_size = dataset['y_{0:s}'.format(group)].shape[0]
	indices = range(dataset_size)
	np.random.shuffle(indices)
	while idx < dataset_size:
		chunk = slice(idx, idx+batch_size)
		chunk = indices[chunk]
		chunk = sorted(chunk)
		idx = idx + batch_size
		yield dataset['X_{0:s}'.format(group)][chunk], dataset['y_{0:s}'.format(group)][chunk]

def main():
    data_set = np.load('gtsrb_dataset.npz')
    generate_motherfucker =batch_generator(data_set, 'train')
    (x, y) = generate_motherfucker.next()
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)


main()