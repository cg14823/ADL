from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf
from cleverhans.model import CallableModelWrapper



here = os.path.dirname(__file__)
sys.path.append(here)
sys.path.append(os.path.join(here, '..', 'CIFAR10'))

CLASS_COUNT  = 43
IMG_WIDTH    = 32
IMG_HEIGHT   = 32
IMG_CHANNELS = 3
BATCH_SIZE   = 100

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('log-frequency', 1,
                            'Number of steps between logging results to the console and saving summaries.' +
                            ' (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 5,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model-frequency', 5,
                            'Number of steps between model saves. (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('max-steps', 100,
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


def deepnn(x_image, class_count=43):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.

    Args:
        x_image: an input tensor whose ``shape[1:] = img_space``
            (i.e. a batch of images conforming to the shape specified in ``img_shape``)
        class_count: number of classes in dataset

    Returns: A tensor of shape (N_examples, 10), with values equal to the logits of
      classifying the object images into one of 10 classes
      (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
    """
    # pad the inputs to the convloution layer
    # padding = tf.constant([2,2],[2,2])
    # pad_image = tf.pad
    # First convolutional layer - maps one RGB image to 32 feature maps.
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05),
        strides=(1,1),
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        use_bias=False,
        name='conv1'
    )

    #Pad again?
    pool1 = tf.layers.average_pooling2d(
        inputs=conv1,
        pool_size=[3, 3],
        strides=(2,2),
        name='pool1',
        padding='same'
    )

    #Pad again?
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05),
        kernel_size=[5, 5],
        padding='same',
        strides=(1,1),
        activation=tf.nn.relu,
        use_bias=False,
        name='conv2'
    )

    #Pad again?
    pool2 = tf.layers.average_pooling2d(
        inputs=conv2,
        pool_size=[3, 3],
        strides=(2,2),
        name='pool2',
        padding='same'
    )

    #Pad again?
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05),
        kernel_size=[5, 5],
        padding='same',
        strides=(1,1),
        activation=tf.nn.relu,
        use_bias=False,
        name='conv3'
    )

    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[3, 3],
        strides=(2,2),
        name='pool3',
        padding='same'
    )

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_initializer=tf.random_uniform_initializer(-0.05,0.05),
        kernel_size=[4, 4],
        padding='valid',
        strides=(1,1),
        activation=tf.nn.relu,
        use_bias=False,
        name='conv4'
    )
    conv4_flat = tf.reshape(conv4, [-1,64], name='conv4_flattened')

    fc1 = tf.layers.dense(inputs=conv4_flat, activation=tf.nn.relu, units=1024, name='fc1')
    logits = tf.layers.dense(inputs=fc1, activation=tf.nn.softmax, units=class_count, name='fc2')
    return logits

def logLoss((logitIn,classIn)):
    val1 = tf.exp(logitIn)
    val2 = tf.reduce_sum(val1)
    val3 = tf.log(val2)
    val4 = logitIn[classIn]
    return tf.subtract(val3,val4)

def main(_):
    tf.reset_default_graph()
    data_set = np.load('gtsrb_dataset.npz')

    # Build the graph for the deep net
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None ,IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
        x_image = tf.map_fn(tf.image.per_image_standardization, x)
        # the tf fucntion above should perform whitening https://www.tensorflow.org/versions/r1.3/api_docs/python/tf/image/per_image_standardization
        y_ = tf.placeholder(tf.float32, shape=[None, CLASS_COUNT])

    with tf.variable_scope('model'):
        logits = deepnn(x_image)
        model = CallableModelWrapper(deepnn, 'logits')
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        maxClass = tf.argmax(y_, 1)
        cross_entropy = tf.reduce_mean(tf.negative(tf.log(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))))
        
        #cross_entropy_temp = tf.subtract(tf.log(tf.reduce_sum(tf.exp(logits)),logits))

        not_cross_entropy = tf.map_fn(logLoss,(logits, maxClass))

        
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
        train_step_temp = optimizer.compute_gradients(cross_entropy)
        train_step = optimizer.apply_gradients(train_step_temp)
        
        
    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    learning_rate_summary = tf.summary.scalar("Learning Rate", learning_rate)
    img_summary = tf.summary.image('Input Images', x_image)

    train_summary = tf.summary.merge([loss_summary, accuracy_summary, learning_rate_summary, img_summary])
    validation_summary = tf.summary.merge([loss_summary, accuracy_summary])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validation", sess.graph)

        sess.run(tf.global_variables_initializer())
        prevValidationAcc = 0
        learningRate = 0.01
        # Training and validation
        for step in range(0, FLAGS.max_steps, 1):
            iteration = 1
            for (train_images, train_labels) in batch_generator(data_set, 'train'):               
                _, train_summary_str,logits_out,not_cross_entropy_out = sess.run([train_step, train_summary,logits,not_cross_entropy],
                                                feed_dict={x: train_images, y_: train_labels, learning_rate: learningRate})
                print('Train Iter {} : '.format(iteration))
                print(not_cross_entropy_out)
                print(np.shape(not_cross_entropy_out))
                print('+----------------------------------+')
                iteration += 1

                # Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                train_writer.add_summary(train_summary_str, step)
                valid_acc_tmp = 0
                validation_steps = 0
                for (test_images, test_labels) in batch_generator(data_set, 'test'):
                    validation_accuracy, validation_summary_str = sess.run([accuracy, validation_summary],
                                                                        feed_dict={x: test_images, y_: test_labels})
                    valid_acc_tmp += validation_accuracy
                    validation_steps += 1
                    validation_writer.add_summary(validation_summary_str, step)
                valid_acc = valid_acc_tmp/validation_steps
                if valid_acc <= prevValidationAcc:
                    learningRate = learningRate/10
                    print('Learning Rate decreased')
                prevValidationAcc = valid_acc
                print('step {}, accuracy on validation set : {}'.format(step, valid_acc))

            # Save the model checkpoint periodically.
            if step % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, checkpoint_path, global_step=step)

            if step % FLAGS.flush_frequency == 0:
                train_writer.flush()
                validation_writer.flush()

        # Resetting the internal batch indexes
        test_batch = batch_generator(data_set,'test')
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

        while evaluated_images != data_set['X_test'].shape[0]:
            # Don't loop back when we reach the end of the test set
            (test_images, test_labels) = test_batch.next()
            temp_acc = sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})
            test_accuracy += temp_acc 
            batch_count += 1
            evaluated_images += test_labels.shape[0]

        test_accuracy = test_accuracy / batch_count

        print('test set: accuracy on test set: %0.3f' % test_accuracy)

        print('model saved to ' + checkpoint_path)

        train_writer.close()
        validation_writer.close()



def batch_generator(dataset, group, batch_size=BATCH_SIZE):

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

if __name__ == '__main__':
    tf.app.run(main=main)