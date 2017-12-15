from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf
import cPickle as pickle
import time
import random
import scipy.ndimage



here = os.path.dirname(__file__)
sys.path.append(here)
sys.path.append(os.path.join(here, '..', 'CIFAR10'))

CLASS_COUNT  = 43
IMG_WIDTH    = 32
IMG_HEIGHT   = 32
IMG_CHANNELS = 3
BATCH_SIZE   = 100
APPLY_RANDOM_BLUR = 0
fgsm_eps = 0.05
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('log-frequency', 100,
                            'Number of steps between logging results to the console and saving summaries.' +
                            ' (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 100,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model-frequency', 100,
                            'Number of steps between model saves. (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('max-steps', 4000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', BATCH_SIZE, 'Number of examples per mini-batch. (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-2, 'Number of examples to run. (default: %(default)d)')

run_log_dir = os.path.join(FLAGS.log_dir,
                           ('exp_bs_{bs}_lr_{lr}_GetMisClass_eps_{eps}')
                           .format(bs=FLAGS.batch_size, lr=FLAGS.learning_rate, eps=fgsm_eps))
checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

def deepnn(x_image,regularizer, class_count=CLASS_COUNT):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.

    Args:
        x_image: an input tensor whose ``shape[1:] = img_space``
            (i.e. a batch of images conforming to the shape specified in ``img_shape``)
        class_count: number of classes in dataset

    Returns: A tensor of shape (N_examples, 43), with values equal to the logits of
      classifying the object images into one of 43 classes
    """
    # pad the inputs to the convloution layer
    # padding = tf.constant([2,2],[2,2])
    # pad_image = tf.pad
    # First convolutional layer - maps one RGB image to 32 feature maps.
    
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        use_bias=False,
        name='conv1',
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05),
        kernel_regularizer=regularizer
    )
    conv1_relu = tf.nn.relu(conv1)

    #Pad again?
    pool1 = tf.layers.average_pooling2d(
        inputs=conv1_relu,
        pool_size=[3, 3],
        strides=2,
        name='pool1',
        padding='same'
    )

    #Pad again?
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        use_bias=False,
        name='conv2',
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05),
        kernel_regularizer=regularizer
    )
    conv2_relu = tf.nn.relu(conv2)

    #Pad again?
    pool2 = tf.layers.average_pooling2d(
        inputs=conv2_relu,
        pool_size=[3, 3],
        strides=2,
        name='pool2',
        padding='same'
    )

    #Pad again?
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        use_bias=False,
        name='conv3',
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05),
        kernel_regularizer=regularizer
    )
    conv3_relu = tf.nn.relu(conv3)

    pool3 = tf.layers.max_pooling2d(
        inputs=conv3_relu,
        pool_size=[3, 3],
        strides=2,
        name='pool3',
        padding='same'
    )

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_size=[4, 4],
        padding='valid',
        use_bias=False,
        name='conv4',
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05),
        kernel_regularizer=regularizer
    )
    conv4_relu = tf.nn.relu(conv4)
    conv4_flat = tf.reshape(conv4_relu, [-1,64], name='conv4_flattened')

    fc1 = tf.layers.dense(inputs=conv4_flat, units=64, name='fc1',kernel_initializer=tf.random_uniform_initializer(-0.05,0.05),kernel_regularizer=regularizer)
    fc1_relu = tf.nn.relu(fc1)
    logits = tf.layers.dense(inputs=fc1_relu, units=class_count, name='fc2',kernel_initializer=tf.random_uniform_initializer(-0.05,0.05),kernel_regularizer=regularizer)
    return (logits,conv1,conv4)

def main(_):
    tf.reset_default_graph()
    data_set = pickle.load(open('dataset.pkl','rb'))

    # Build the graph for the deep net
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32,shape=[None ,IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
        x_image = tf.map_fn(tf.image.per_image_standardization, x)
        
        #x_image = x

        # the tf fucntion above should perform whitening https://www.tensorflow.org/versions/r1.3/api_docs/python/tf/image/per_image_standardization
        y_ = tf.placeholder(tf.float32, shape=[None, CLASS_COUNT])

    with tf.variable_scope('model'):
        learning_rate = tf.placeholder(tf.float32, shape=[])
        def momentumReg(weights):
            return tf.subtract(weights,tf.scalar_mul(tf.scalar_mul(0.0001,learning_rate),weights))
        (logits,fc1,conv4_flat) = deepnn(x_image,momentumReg)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        def logLoss(logitIn,classTen):
            val5 = tf.argmax(classTen)
            val1 = tf.exp(logitIn)
            val2 = tf.reduce_sum(val1)
            val3 = tf.log(val2)
            val6 = tf.gather(logitIn,val5)
            val3 = tf.cond(tf.is_finite(val3),lambda: val3,lambda: tf.add(val6,tf.constant(0.001))) 
            return tf.subtract(val3,val6)

        not_cross_entropy = tf.map_fn(lambda (v1,v2):logLoss(v1,v2),(logits,y_),dtype=tf.float32)

        our_loss = tf.reduce_mean(not_cross_entropy)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        error = tf.subtract(tf.constant(1,dtype=tf.float32),accuracy)
        
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum = 0.9)  
        train_step = optimizer.minimize(our_loss)
        
    
        
    loss_summary = tf.summary.scalar("Loss", our_loss)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    error_summary = tf.summary.scalar("Error", error)
    learning_rate_summary = tf.summary.scalar("Learning Rate", learning_rate)
    img_summary = tf.summary.image('Input Images', x_image)
    in_summary = tf.summary.image('Pre Whitening Images', x)
    kernel_images_1_in = tf.placeholder(tf.float32)
    kernel_images_2_in = tf.placeholder(tf.float32)
    kernel_img_summary_1 = tf.summary.image('Kernel Images', kernel_images_1_in,32)
    kernel_img_summary_2 = tf.summary.image('Kernel 2 Images', kernel_images_2_in,32)
    mis_class_imgs = tf.zeros(dtype=tf.float32,shape=[BATCH_SIZE ,IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
    ind = 0
    def fnHere(iVal):
        mis_class_imgs[ind,:,:,:] = x[iVal,:,:,:]
        ind+=1
    for i in range(100):
        tf.cond((tf.equal(tf.argmax(logits[i,:],0),tf.argmax(y_[i,:],0))),lambda:fnHere(i),lambda:pass)            
    mis_classified_img_summary = tf.summary.image('Mis-Classified Images', mis_class_imgs,32)

    train_summary = tf.summary.merge([loss_summary, accuracy_summary, learning_rate_summary,in_summary, img_summary,error_summary])
    validation_summary = tf.summary.merge([loss_summary, accuracy_summary,error_summary])
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validation", sess.graph)

        sess.run(tf.global_variables_initializer())
        prevValidationAcc = []
        stdCheck = 0.017
        learningRate = 0.01
        # Training and validation
        step = 0
        epoch = 0
        while step < FLAGS.max_steps:
            
            for (train_images, train_labels) in batch_generator(data_set, 'train'):  
                if step > FLAGS.max_steps:
                    break
                if (APPLY_RANDOM_BLUR):
                    for i in range(len(train_images)):
                        if (random.randint(0,2) == 0):
                            train_images[i] = applyMotionBlur(train_images[i])
                _, train_summary_str = sess.run([train_step, train_summary],
                                                feed_dict={x: train_images, y_: train_labels, learning_rate: learningRate})
                if step > 0:
                    if step % FLAGS.log_frequency == 0:
                        train_writer.add_summary(train_summary_str, step)

                    # Validation: Monitoring accuracy using validation set
                    if step % FLAGS.log_frequency == 0:
                        valid_acc_tmp = 0
                        validation_steps = 0
                        for (test_images, test_labels) in batch_generator(data_set, 'test'):
                            validation_accuracy, validation_summary_str = sess.run([accuracy, validation_summary],
                                                                                feed_dict={x: test_images, y_: test_labels, learning_rate: learningRate})
                            valid_acc_tmp += validation_accuracy
                            validation_steps += 1
                            validation_writer.add_summary(validation_summary_str, step)
                        valid_acc = valid_acc_tmp/validation_steps
                        prevValidationAcc = [valid_acc] + prevValidationAcc
                        if epoch >= 3:
                            prevValidationAcc.pop()
                            if (np.std(prevValidationAcc) < stdCheck and epoch > 5):
                                learningRate = learningRate/10
                                stdCheck = stdCheck/5
                                print('Learning Rate decreased to : {}, stdCheck = {}'.format(learningRate,stdCheck))
                            print(np.std(prevValidationAcc))
                            
                        
                        print('Step {}, Epoch {}, accuracy on validation set : {}'.format(step,epoch, valid_acc))
                        epoch += 1
                    
                    # Save the model checkpoint periodically.
                    if step % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                        saver.save(sess, checkpoint_path, global_step=step)

                    if step % FLAGS.flush_frequency == 0:
                        train_writer.flush()
                        validation_writer.flush()
                step += 1
        # Resetting the internal batch indexes
        '''
        kernel_writer = tf.summary.FileWriter(run_log_dir + "_kernels", sess.graph)
        gr = tf.get_default_graph()
        conv1_kernel = gr.get_tensor_by_name('model/conv1/kernel:0').eval()
        conv1_kernel_in = np.zeros([32,5,5,3])
        conv2_kernel_in = np.zeros([32,5,5,1])
        for i in range(0,32):
            conv1_kernel_in[i,:,:,:] = conv1_kernel[:,:,:,i]
            conv2_kernel_in[i,:,:,0] = conv2_kernel[:,:,0,i]
        [kernel_sum_out,kernel_sum_out_2]= sess.run([kernel_img_summary_1,kernel_img_summary_2], feed_dict={kernel_images_1_in: conv1_kernel_in,kernel_images_2_in: conv2_kernel_in})
        kernel_writer.add_summary(kernel_sum_out)
        kernel_writer.add_summary(kernel_sum_out_2)
        kernel_writer.flush()
        kernel_writer.close()
        '''
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        correctPredictionCount = [0, 0, 0, 0, 0, 0]
        trueClassCount = [0, 0, 0, 0, 0, 0]
        mis_class_imgs_writer = tf.summary.FileWriter(run_log_dir + "_mis_classed", sess.graph)

        for (test_images, test_labels) in batch_generator(data_set, 'test'):
            temp_acc,logits_out,m_c_i_s_o = sess.run([accuracy,logits,mis_classified_img_summary], feed_dict={x: test_images, y_: test_labels, learning_rate: learningRate})
            test_accuracy += temp_acc 
            batch_count += 1
            correctPredictionCount, trueClassCount = evaluatePerClass(correctPredictionCount, trueClassCount, logits_out, test_labels)
            evaluated_images += np.shape(test_labels)[0]
            mis_class_imgs_writer.add_summary(m_c_i_s_o,batch_count)

        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        mis_class_imgs_writer.flush()
        accuracyPerClass = [0.0,0.0,0.0,0.0,0.0,0.0]
        for i in range(6):
            accuracyPerClass[i] = float(correctPredictionCount[i]) / trueClassCount[i]

        print("Accuracy per class")
        print("O: {:.4f} 1:{:.4f} 2:{:.4f} 3:{:.4f} 4:{:.4f} 5:{:.4f}".format(accuracyPerClass[0], accuracyPerClass[1], accuracyPerClass[2], accuracyPerClass[3], accuracyPerClass[4], accuracyPerClass[5]))
        print('model saved to ' + checkpoint_path)
        train_writer.close()
        validation_writer.close()
        mis_class_imgs_writer.close()

def applyRandomBlur(imageIn):
    imageOut = scipy.ndimage.filters.gaussian_filter(imageIn,2)
    return imageOut

def applyMotionBlur(imageIn):
    zA = [0,0,0]
    oA = [1,1,1]
    tA = [2,2,2]
    motionBlurKernel = np.array([[zA, zA, zA, zA, zA],[zA, zA, zA, zA, zA],[oA, oA, tA, oA, oA],[zA, zA, zA, zA, zA],[zA, zA, zA, zA, zA]])
    motionBlurKernel = np.divide(motionBlurKernel,18.)
    imageOut = scipy.ndimage.filters.convolve(imageIn,motionBlurKernel,mode='nearest',cval=0.0)
    imageOut = np.clip(imageOut,0,1)
    return imageOut

def evaluatePerClass(correctPreditionCount, count, logits, labels):
    trueClass = np.argmax(labels,1)
    predClass = np.argmax(logits,1)
    
    count[0] += np.count_nonzero(trueClass == 0) + np.count_nonzero(trueClass == 1) + np.count_nonzero(trueClass == 2) + np.count_nonzero(trueClass == 3) + np.count_nonzero(trueClass == 4) + np.count_nonzero(trueClass == 5) + np.count_nonzero(trueClass == 7) + np.count_nonzero(trueClass == 8) 
    count[1] += np.count_nonzero(trueClass == 9) + np.count_nonzero(trueClass == 10) + np.count_nonzero(trueClass == 15) + np.count_nonzero(trueClass == 16)
    count[2] += np.count_nonzero(trueClass == 6) + np.count_nonzero(trueClass == 32) + np.count_nonzero(trueClass == 41) + np.count_nonzero(trueClass == 42)
    count[3] += np.count_nonzero(trueClass == 33) + np.count_nonzero(trueClass == 34) + np.count_nonzero(trueClass == 35) + np.count_nonzero(trueClass == 36) + np.count_nonzero(trueClass == 37) + np.count_nonzero(trueClass == 38) + np.count_nonzero(trueClass == 39) + np.count_nonzero(trueClass == 40)
    count[4] += np.count_nonzero(trueClass == 11) + np.count_nonzero(trueClass == 18) + np.count_nonzero(trueClass == 19) + np.count_nonzero(trueClass == 20) + np.count_nonzero(trueClass == 21) + np.count_nonzero(trueClass == 22) + np.count_nonzero(trueClass == 23) + np.count_nonzero(trueClass == 24) + np.count_nonzero(trueClass == 25) + np.count_nonzero(trueClass == 26) + np.count_nonzero(trueClass == 27) + np.count_nonzero(trueClass == 28) + np.count_nonzero(trueClass == 29) + np.count_nonzero(trueClass == 30) + np.count_nonzero(trueClass == 31)
    count[5] += np.count_nonzero(trueClass == 12) + np.count_nonzero(trueClass == 13) + np.count_nonzero(trueClass == 14) + np.count_nonzero(trueClass == 17)


    i = 0
    for x in predClass:
        if x == trueClass[i]:
            if x < 6 or x == 7  or x == 8:
                correctPreditionCount[0] += 1
            elif x == 9 or x == 10 or x == 15 or x == 16:
                correctPreditionCount[1] += 1
            elif x == 6 or x == 32 or x == 41 or x == 42:
                correctPreditionCount[2] += 1
            elif x > 32 and x < 41:
                correctPreditionCount[3] += 1
            elif x == 11 or (x > 17 and x < 32):
                correctPreditionCount[4] += 1
            elif x == 12 or x == 13 or x == 14 or x == 17:
                correctPreditionCount[5] += 1
            else:
                print("SHIIIIT x: {}".format(x))
        i += 1

    return correctPreditionCount, count

def batch_generator(dataset, group, batch_size=BATCH_SIZE):

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

if __name__ == '__main__':
    tf.app.run(main=main)
