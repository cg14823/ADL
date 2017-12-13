from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf
import cPickle as pickle



here = os.path.dirname(__file__)
sys.path.append(here)
sys.path.append(os.path.join(here, '..', 'CIFAR10'))

CLASS_COUNT  = 43
IMG_WIDTH    = 32
IMG_HEIGHT   = 32
IMG_CHANNELS = 3
BATCH_SIZE   = 100

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
tf.app.flags.DEFINE_integer('max-steps', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', BATCH_SIZE, 'Number of examples per mini-batch. (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-2, 'Number of examples to run. (default: %(default)d)')


fgsm_eps = 0.05
adversarial_training_enabled = False
run_log_dir = os.path.join(FLAGS.log_dir,
                           ('exp_bs_{bs}_lr_{lr}_' + ('adv_trained' if adversarial_training_enabled else '') + 'eps_{eps}')
                           .format(bs=FLAGS.batch_size, lr=FLAGS.learning_rate, eps=fgsm_eps))
checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

def weight_variable(shape):
    '''weight_variable generates a weight variable of a given shape.'''
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial, name='weights')

def deepnn(x_image, class_count=43):
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
    '''
    with tf.variable_scope('Conv_1'):
        W_conv1 = weight_variable([5, 5, 3, 32])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='conv1'))

        # Pooling layer - downsamples by 2X.
        h_pool1 = tf.nn.avg_pool(h_conv1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pooling1')

    with tf.variable_scope('Conv_2'):
        # Second convolutional layer -- maps 32 feature maps to 32.
        W_conv2 = weight_variable([5, 5, 32, 32])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME', name='conv2'))

        # Second pooling layer.
        h_pool2 = tf.nn.avg_pool(h_conv2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pooling2')

    with tf.variable_scope('Conv_3'):
        # Second convolutional layer -- maps 32 feature maps to 64.
        W_conv3 = weight_variable([5, 5, 32, 64])
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME', name='conv3'))

        # Second pooling layer.
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pooling3')

    with tf.variable_scope('Conv_4'):
        # Second convolutional layer -- maps 32 feature maps to 64.
        W_conv4 = weight_variable([4, 4, 64, 64])
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4, strides=[1, 1, 1, 1], padding='VALID', name='conv4'))


    with tf.variable_scope('Conv_4'):
        # Second convolutional layer -- maps 64 feature maps to 64.
        h_conv4 = tf.layers.dense(inputs=h_pool3, activation=tf.nn.relu,units= 64, name='conv4')
    
    with tf.variable_scope('FC_1'):
        h_conv4_flat = tf.reshape(h_conv4,[-1,64])
        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28
        # image is down to 8x8x64 feature maps -- maps this to 1024 features.
        logits = tf.layers.dense(inputs=h_conv4_flat,units = 43,name='fc1')
        return logits

    '''
    
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        use_bias=False,
        name='conv1',
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05)
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
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05)
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
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05)
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
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05)
    )
    conv4_relu = tf.nn.relu(conv4)
    conv4_flat = tf.reshape(conv4_relu, [-1,64], name='conv4_flattened')

    fc1 = tf.layers.dense(inputs=conv4_flat, activation=tf.nn.relu, units=64, name='fc1',kernel_initializer=tf.random_uniform_initializer(-0.05,0.05))
    logits = tf.layers.dense(inputs=fc1, units=class_count, name='fc2',kernel_initializer=tf.random_uniform_initializer(-0.05,0.05))
    return (logits,conv1,conv4)



def main(_):
    tf.reset_default_graph()
    data_set = pickle.load(open('dataset.pkl','rb'))

    # Build the graph for the deep net
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None ,IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
        x_image = tf.map_fn(tf.image.per_image_standardization, x)
        # the tf fucntion above should perform whitening https://www.tensorflow.org/versions/r1.3/api_docs/python/tf/image/per_image_standardization
        y_ = tf.placeholder(tf.float32, shape=[None, CLASS_COUNT])

    with tf.variable_scope('model'):
        (logits,fc1,conv4_flat) = deepnn(x_image)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        def logLoss(logitIn,classTen):
            val5 = tf.argmax(classTen)
            val1 = tf.exp(logitIn)
            val2 = tf.reduce_sum(val1)
            val3 = tf.log(val2)
            val6 = tf.gather(logitIn,val5)
            val3 = tf.cond(tf.is_finite(val3),lambda: val3,lambda: tf.add(val6,tf.constant(0.001))) 
            return tf.subtract(val3,val6)
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

        not_cross_entropy = tf.map_fn(lambda (v1,v2):logLoss(v1,v2),(logits,y_),dtype=tf.float32)

        our_loss = tf.reduce_mean(not_cross_entropy)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        '''
        decay_steps = 1000  # decay the learning rate every 1000 steps
        decay_rate = 0.0001  # the base of our exponential for the decay
        global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow
        decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                           decay_steps, decay_rate, staircase=True)
        train_step = tf.train.MomentumOptimizer(decayed_learning_rate, 0.9).minimize(cross_entropy, global_step=global_step)
        '''
        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum = 0.9)
        out_grad = optimizer.compute_gradients(our_loss)
        train_step = optimizer.apply_gradients(out_grad)
        #train_step_temp = optimizer.compute_gradients(our_loss)
        #train_step = optimizer.apply_gradients(train_step_temp)
        
        
    loss_summary = tf.summary.scalar("Loss", our_loss)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    learning_rate_summary = tf.summary.scalar("Learning Rate", learning_rate)
    img_summary = tf.summary.image('Input Images', x_image)
    in_summary = tf.summary.image('Pre Whitening Images', x)

    train_summary = tf.summary.merge([loss_summary, accuracy_summary, learning_rate_summary,in_summary, img_summary])
    test_summary = tf.summary.merge([loss_summary,accuracy_summary,in_summary,img_summary])
    validation_summary = tf.summary.merge([loss_summary, accuracy_summary])
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validation", sess.graph)

        sess.run(tf.global_variables_initializer())
        prevValidationAcc = []
        learningRate = 0.01
        # Training and validation
        step = 0
        epoch = 0
        while step < FLAGS.max_steps:
            
            for (train_images, train_labels) in batch_generator(data_set, 'train'):  
                _, train_summary_str,our_loss_out,not_cross_entropy_out,accuracy_out,out_grad_out = sess.run([train_step, train_summary,our_loss,not_cross_entropy,accuracy,out_grad],
                                                feed_dict={x: train_images, y_: train_labels, learning_rate: learningRate})
                '''
                if step > 1500:
                    print("Step {} ============".format(step))
                    print(accuracy_out)
                    print(our_loss_out)
                    print(not_cross_entropy_out)
                '''
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
                    if epoch >= 3:
                        prevValidationAcc.pop()
                        if (np.std(prevValidationAcc) < 0.01):
                            learningRate = learningRate/10
                            print('Learning Rate decreased to : {}'.format(learningRate))
                        print(np.std(prevValidationAcc))
                        
                    prevValidationAcc = [valid_acc] + prevValidationAcc
                    print('Step {}, Epoch {}, accuracy on validation set : {}'.format(step,epoch, valid_acc))
                    epoch += 1
                
                # Save the model checkpoint periodically.
                if step % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                    saver.save(sess, checkpoint_path, global_step=step)

                if step % FLAGS.flush_frequency == 0:
                    train_writer.flush()
                    validation_writer.flush()
                #if valid_acc > 0.9:
                    #step = FLAGS.max_steps
                    #break
                step += 1
        # Resetting the internal batch indexes
        test_batch = batch_generator(data_set,'test')
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        test_writer = tf.summary.FileWriter(run_log_dir + "_test", sess.graph)
        for (test_images, test_labels) in batch_generator(data_set, 'test'):
            temp_acc,test_sum_out = sess.run([accuracy,test_summary], feed_dict={x: test_images, y_: test_labels, learning_rate: learningRate})
            test_writer.add_summary(test_sum_out, batch_count)
            test_accuracy += temp_acc 
            batch_count += 1
            evaluated_images += np.shape(test_labels)[0]

        test_accuracy = test_accuracy / batch_count
        test_writer.flush()
        print('test set: accuracy on test set: %0.3f' % test_accuracy)

        print('model saved to ' + checkpoint_path)
        test_writer.close()
        train_writer.close()
        validation_writer.close()

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
