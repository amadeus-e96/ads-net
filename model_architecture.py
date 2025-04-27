import keras
from keras.layers import Lambda, Dropout
from keras import backend as K
import tensorflow as tf
from keras.layers import concatenate, Activation, BatchNormalization, Conv2D, MaxPooling2D

def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(1-tf.image.ssim(y_true, y_pred, 2.0))

def conv_batchnorm_relu_block(input_tensor, nb_filter, kernel_size=3):

    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same')(input_tensor)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)

    return x

def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(1-tf.image.ssim(y_true, y_pred, 2.0))

def SMAPE(y_true, y_pred):
  return tf.reduce_mean(abs(y_true - y_pred)/(abs(y_true)+abs(y_pred)+0.01))
 
def ssimSMAPE_loss(y_true, y_pred):
  return tf.reduce_mean(0.8*(1-tf.image.ssim(y_true, y_pred, 1.0))+0.2*(abs(y_true - y_pred)/(abs(y_true)+abs(y_pred)+0.01)))

def SubpixelConv2D(input_shape,names, scale=4):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        import tensorflow as tf
        return tf.nn.depth_to_space(x, scale)


    return Lambda(subpixel, output_shape=subpixel_shape, name=names)


def model_ADSNET(inputs, n_labels, dropout=0.1, using_deep_supervision=False):

    nb_filter = [16,32,64,128,256]

    global bn_axis

    K.set_image_data_format("channels_last")
    bn_axis = -1

    conv1_1 = conv_batchnorm_relu_block(inputs, nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2))(conv1_1)
    pool1 = Dropout(dropout)(pool1)
    
    conv2_1 = conv_batchnorm_relu_block(pool1, nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2))(conv2_1)
    pool2 = Dropout(dropout)(pool2)

    up1_2 = SubpixelConv2D(conv2_1,'up12',scale=2)(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = conv_batchnorm_relu_block(conv1_2,  nb_filter=nb_filter[0])
    conv1_2 = Dropout(dropout)(conv1_2)

    conv3_1 = conv_batchnorm_relu_block(pool2, nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2))(conv3_1)
    pool3 = Dropout(dropout)(pool3)

    up2_2 = SubpixelConv2D(conv3_1,'up22',scale=2)(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = conv_batchnorm_relu_block(conv2_2, nb_filter=nb_filter[1])
    conv2_2 = Dropout(dropout)(conv2_2)

    up1_3 = SubpixelConv2D(conv2_2,'up13',scale=2)(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = conv_batchnorm_relu_block(conv1_3, nb_filter=nb_filter[0])
    conv1_3 = Dropout(dropout)(conv1_3)

    conv4_1 = conv_batchnorm_relu_block(pool3, nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2))(conv4_1)
    pool4 = Dropout(dropout)(pool4)

    up3_2 = SubpixelConv2D(conv4_1,'up32',scale=2)(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = conv_batchnorm_relu_block(conv3_2, nb_filter=nb_filter[2])
    conv3_2 = Dropout(dropout)(conv3_2)

    up2_3 = SubpixelConv2D(conv3_2,'up23',scale=2)(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = conv_batchnorm_relu_block(conv2_3, nb_filter=nb_filter[1])
    conv2_3 = Dropout(dropout)(conv2_3)

    up1_4 = SubpixelConv2D(conv2_3,'up14',scale=2)(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = conv_batchnorm_relu_block(conv1_4, nb_filter=nb_filter[0])
    conv1_4 = Dropout(dropout)(conv1_4)

    conv5_1 = conv_batchnorm_relu_block(pool4, nb_filter=nb_filter[4])
    conv5_1 = Dropout(dropout)(conv5_1)

    up4_2 = SubpixelConv2D(conv5_1,'up42',scale=2)(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = conv_batchnorm_relu_block(conv4_2, nb_filter=nb_filter[3])
    conv4_2 = Dropout(dropout)(conv4_2)

    up3_3 = SubpixelConv2D(conv4_2,'up33',scale=2)(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = conv_batchnorm_relu_block(conv3_3, nb_filter=nb_filter[2])
    conv3_3 = Dropout(dropout)(conv3_3)

    up2_4 = SubpixelConv2D(conv3_3,'up24',scale=2)(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = conv_batchnorm_relu_block(conv2_4, nb_filter=nb_filter[1])
    conv2_4 = Dropout(dropout)(conv2_4)

    up1_5 = SubpixelConv2D(conv2_4,'up15',scale=2)(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = conv_batchnorm_relu_block(conv1_5, nb_filter=nb_filter[0])
    conv1_5 = Dropout(dropout)(conv1_5)

    nestnet_output_1 = Conv2D(n_labels, (1, 1), activation='linear', name='output_1',padding='same')(conv1_2)
    nestnet_output_2 = Conv2D(n_labels, (1, 1), activation='linear', name='output_2', padding='same' )(conv1_3)
    nestnet_output_3 = Conv2D(n_labels, (1, 1), activation='linear', name='output_3', padding='same')(conv1_4)
    nestnet_output_4 = Conv2D(n_labels, (1, 1), activation='linear', name='output_4', padding='same')(conv1_5)

    conv1_6 = concatenate([nestnet_output_1,nestnet_output_2, nestnet_output_3, nestnet_output_4], name='merge16', axis=bn_axis)
    nestnet_output_5 = Conv2D(n_labels, (1, 1), activation='linear', name='output_5',padding='same')(conv1_6)
    nestnet_output_5 = Dropout(dropout)(nestnet_output_5)
    
    if using_deep_supervision:
        model = keras.Model(input=inputs, output=[nestnet_output_1,
                                            nestnet_output_2,
                                            nestnet_output_3,
                                            nestnet_output_4])
    else:
        model = keras.Model(inputs=inputs, outputs=nestnet_output_5)

    return model