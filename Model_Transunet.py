import os

import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint
from Data import *


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = Dropout(0.1)(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    return x


def dilated_conv_block(inputs, num_filters, dilation_rate):
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               dilation_rate=dilation_rate)(inputs)
    x = Dropout(0.1)(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
               dilation_rate=dilation_rate)(x)
    return x


def trans_conv_block(inputs, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    return x


def Model_TransUnet3Plus(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(input_shape)

    conv1 = dilated_conv_block(inputs, 64, 1)
    conv1 = conv_block(conv1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = dilated_conv_block(pool1, 128, 2)
    conv2 = conv_block(conv2, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = dilated_conv_block(pool2, 256, 4)
    conv3 = conv_block(conv3, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = dilated_conv_block(pool3, 512, 8)
    conv4 = conv_block(conv4, 512)
    drop4 = Dropout(0.5)(conv4)

    up5 = trans_conv_block(drop4, 256)
    up5 = concatenate([up5, conv3])
    conv5 = dilated_conv_block(up5, 256, 4)
    conv5 = conv_block(conv5, 256)

    up6 = trans_conv_block(conv5, 128)
    up6 = concatenate([up6, conv2])
    conv6 = dilated_conv_block(up6, 128, 2)
    conv6 = conv_block(conv6, 128)

    up7 = trans_conv_block(conv6, 64)
    up7 = concatenate([up7, conv1])
    conv7 = dilated_conv_block(up7, 64, 1)
    conv7 = conv_block(conv7, 64)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv7)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def Model_TransUnet(Images, sol=None):
    if sol is None:
        sol = [4, 5, 0.01, 0.01]
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    myGene = trainGenerator(Images, 'Images', 'Masks', data_gen_args, save_to_dir=None)

    image_list = os.listdir(Images)
    image_count = len(image_list)

    testGene = testGenerator(Images, num_image=image_count)

    model = Model_TransUnet3Plus()
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    if sol:
        model.fit_generator(myGene, steps_per_epoch=300, batch_size=sol[0], layers=sol[1],
                            learning_rate=sol[2], dropout=sol[3], epochs=100, callbacks=[model_checkpoint])
    else:
        model.fit_generator(myGene, steps_per_epoch=1500, epochs=100, callbacks=[model_checkpoint])
    results = model.predict_generator(testGene, image_count, verbose=1)
    Images = saveResult(Images + "Predict/", results)
    return Images
