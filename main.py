"""
Our Fully Convolutional Network is using the FCN-8 architecture as described in:
 https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

The encoder is based on VGG-16:
 https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip
 The VGG-16 layer dimensions are:  
  (https://www.cs.toronto.edu/~frossard/post/vgg16/)
  (https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt)

We train and test using the Kitti Road dataset: 
 http://www.cvlibs.net/download.php?file=data_road.zip
 http://www.cvlibs.net/datasets/kitti/eval_road.php 
"""

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Hyper-parameters
EPOCHS        = 15
BATCH_SIZE    = 24     # I got best result with batch size of 24, 
#                     # but needed to run on AWS g3x4.large instance
KEEP_PROB     = 0.5   # Use only during training
REG           = 1.E-2 # For regularization

# Hyper-parameters to drive Adam optimizer
LEARNING_RATE = 1.E-4 # Choose value between 1E-4 and 1.E-2


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # first, load the model from file into our tensorflow session
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # then, extract those tensors from the model that we will use in extending vgg to our FCN
    graph = tf.get_default_graph()
    input_image    = graph.get_tensor_by_name(vgg_input_tensor_name)   # get the input layer
    keep_prob      = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    # return the tensors
    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

print("-------------------------------------------")
print("Testing load_vgg")
tests.test_load_vgg(load_vgg, tf)
print("-------------------------------------------")

def conv2d_1x1(layer, num_classes, name):
    init = tf.truncated_normal_initializer(stddev = 0.01)    
    kernel_size=1
    stride     =1    
    return tf.layers.conv2d(layer, num_classes, kernel_size, stride, 
                            padding='same',
                            kernel_initializer = init,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(REG),
                            name=name)

def deconvolute(layer, num_classes, kernel_size, stride, name):
    init = tf.truncated_normal_initializer(stddev = 0.01)    
    return tf.layers.conv2d_transpose(layer, num_classes, kernel_size, stride,
                                    padding='same',
                                    kernel_initializer = init,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(REG),
                                    name=name)

def skip(output_layer, skip_layer, num_classes, kernel_size, stride, name):
    output = deconvolute(output_layer, num_classes, kernel_size, stride, name=name+'_deconv') 
    output = tf.layers.batch_normalization(output)
    return tf.add(output, skip_layer, name=name)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    #
    # Uncomment these to see dimensions of the layers 
    #
    #tf.Print(vgg_layer3_out, [tf.shape(vgg_layer3_out)], message="Shape of vgg_layer3_out: ")
    #tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)], message="Shape of vgg_layer4_out: ")
    #tf.Print(vgg_layer7_out, [tf.shape(vgg_layer7_out)], message="Shape of vgg_layer7_out: ")
    #

    # TODO: Implement function
    
    # transform to depth=num_classes for layers 3,4,7 using a 1x1 convolution
    l3_1x1 = conv2d_1x1(vgg_layer3_out, num_classes, 'l3_1x1')
    l4_1x1 = conv2d_1x1(vgg_layer4_out, num_classes, 'l4_1x1')
    l7_1x1 = conv2d_1x1(vgg_layer7_out, num_classes, 'l7_1x1')
        
    # upsample with two skip connection and a deconvolution back to original image size
    output = skip(l7_1x1, l4_1x1, num_classes,  4, 2, name='skip4')
    output = skip(output, l3_1x1, num_classes,  4, 2, name='skip3')
    output = deconvolute(output , num_classes, 16, 8, name='output') 
    
    return output

print("-------------------------------------------")
print("Testing layers")
tests.test_layers(layers)
print("-------------------------------------------")

def optimize(nn_last_layer, tf_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param tf_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    # reshape output tensor into a 2D tensor where each row represents a pixel
    # and each column a class.
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    
    # then apply softmax to get probability distribution over the classes for each 
    # pixel, and calculate the average cross entropy loss over all pixels
    labels = tf.reshape(tf_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                        labels=labels,logits=logits,name='loss'))
    
    # training is done with Adam optimizer
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss

print("-------------------------------------------")
print("Testing optimize")
tests.test_optimize(optimize)
print("-------------------------------------------")


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             tf_label, keep_prob, learning_rate,
             logits,
             tf_prediction, tf_metric, tf_metric_update, running_vars, running_vars_initializer):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param tf_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param logits: TF Placeholder for logits
    :param tf_prediction: TF Placeholder for predictions of images
    :param tf_metric: metric of a tf.metrics.mean_iou
    :param tf_metric_update: update operation of a tf.metrics.mean_iou 
    :param running_vars: variables used by TF by the metric operation 
    :param running_vars_initializer: TF initalizer for the running_vars
    """
    
    # write the loss & score to csv files, for plotting
    f_loss  = open('./loss.csv', 'w')
    f_score = open('./score.csv', 'w')

    step = 0
    for i in range(epochs):
        count      = 0
        loss_total = 0.0        
        score_total= 0.0
        for images, labels in get_batches_fn(batch_size):
            # Reset the running variables used by iou metric calculation
            sess.run(running_vars_initializer)
                
            # train on this batch
            _, loss_batch = \
                sess.run([train_op, cross_entropy_loss],
                         feed_dict={input_image: images, 
                                    tf_label: labels,
                                    keep_prob:KEEP_PROB, 
                                    learning_rate:LEARNING_RATE})
            

            # calculate the score (IoU) predicted for this batch
            logits_batch = sess.run( [logits], # get predicted logits for this batch   
                                   feed_dict={keep_prob: 1.0, 
                                              input_image: images}) 
            
            logits_batch = logits_batch[0]                             # 0th item is the numpy array of logits
            predictions  = (logits_batch > 0.5)                        # For each pixel of every image for each class, set to True or False
            predictions  = predictions.reshape(labels.shape)           # Get it in correct shape
            
            sess.run(tf_metric_update, 
                     feed_dict={tf_label: labels,
                                tf_prediction: predictions})                                                           # -> This is the format of the labels to compare too
            score = sess.run(tf_metric)                            

            # print loss & IoU for this batch
            print("EPOCH {0:<12d}, BATCH {1:<12d}: Training Loss, IoU score = {2:.3f}, {3:.3f}".format(i+1, count, loss_batch, score))

            loss_total += loss_batch
            score_total+= score
            count += 1
            
            # write loss & score of each batch to a file for plotting
            f_loss.write( "{0:<12d}, {1:.3f}\n".format(step,loss_batch))
            f_loss.flush()
            os.fsync(f_loss.fileno())
            
            f_score.write("{0:<12d}, {1:.3f}\n".format(step,score))
            f_score.flush()
            os.fsync(f_score.fileno())            
            
            
            step += 1
            
            
        loss_epoch = loss_total/count
        score_epoch= score_total/count
        print("EPOCH {0:<12d}: Average loss and IoU score per batch = {1:.3f}, {2:.3f}".format(i+1, loss_epoch, score_epoch))            

print("-------------------------------------------")
print("Testing train_nn")            
tests.test_train_nn(train_nn)
print("-------------------------------------------")

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    
    # Placeholder for learning rate
    tf_learning_rate = tf.placeholder(tf.float32)    

    # Placeholders to take in batches of data
    #tf_label      = tf.placeholder(tf.float32, [None, None, None, num_classes], name='label')
    #tf_prediction = tf.placeholder(tf.float32, [None, None, None, num_classes], name='prediction')
    tf_label      = tf.placeholder(tf.bool, [None, None, None, num_classes], name='label')
    tf_prediction = tf.placeholder(tf.bool, [None, None, None, num_classes], name='prediction')    

    # Define the metric and update operations
    # see: http://ronny.rest/blog/post_2017_09_11_tf_metrics/
    tf_metric, tf_metric_update = tf.metrics.mean_iou(tf_label, tf_prediction, num_classes,
                                                      name="my_metric")

    # Isolate the variables stored behind the scenes by the metric operation
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")

    # Define initializer to initialize/reset running variables
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    
    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layers_output, tf_label, tf_learning_rate, num_classes)
        
        sess.run(tf.global_variables_initializer())
        
        # initialize/reset the running variables
        sess.run(running_vars_initializer)        
        
        # TODO: Train NN using the train_nn function
        epochs        = EPOCHS
        batch_size    = BATCH_SIZE
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 tf_label, keep_prob, tf_learning_rate,
                 logits,
                 tf_prediction, tf_metric, tf_metric_update, running_vars, running_vars_initializer)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        
if __name__ == '__main__':
    run()
