import numpy as np
import cv2 as cv
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D
from keras.layers import Attention
from Evaluation import evaluation


def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)
    return trans_x, tran_filters

def dilated_dense_block(x, num_layers=4, growth_rate=32):
    """Dilated Dense block composed of multiple conv blocks."""
    for _ in range(num_layers):
        x = conv_layer(x, growth_rate)
    return x

def multi_head_attention(inputs, num_heads=4):
    """Multi-Head Attention mechanism."""
    _, H, W, C = inputs.shape
    head_dim = C // num_heads

    # Linear projections for Q, K, V
    Q = Conv2D(C, kernel_size=1)(inputs)
    K = Conv2D(C, kernel_size=1)(inputs)
    V = Conv2D(C, kernel_size=1)(inputs)

    # Reshape for multi-head attention
    Q = tf.reshape(Q, [-1, H, W, num_heads, head_dim])
    K = tf.reshape(K, [-1, H, W, num_heads, head_dim])
    V = tf.reshape(V, [-1, H, W, num_heads, head_dim])

    # Compute attention scores
    attention_scores = tf.einsum('ijklm,ijolm->ijokm', Q, K) / tf.math.sqrt(tf.cast(head_dim, tf.float32))
    attention_weights = tf.nn.softmax(attention_scores, axis=3)

    # Apply attention to value
    attention_output = tf.einsum('ijklm,ijolm->ijklm', attention_weights, V)
    attention_output = tf.reshape(attention_output, [-1, H, W, C])

    return attention_output

def dense_net(num_of_class=1):
    dense_block_size = 3
    layers_in_block = 4
    num_heads = 4
    growth_rate = 12
    num_layers_per_block = 4
    filters = growth_rate * 2
    input_img = Input(shape=(32, 32, 3))
    x1 = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)
    x2 = Conv2D(24, (5, 5), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)
    x3 = Conv2D(24, (7, 7), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)
    x = x1 + x2 + x3 / 3
    dense_x = Activation('relu')(x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)
        dense_x = multi_head_attention(dense_x, num_heads)
        dense_x = dilated_dense_block(dense_x, num_layers_per_block, growth_rate)

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(num_of_class, activation='softmax')(dense_x)
    model = Model(input_img, output)

    return model


def Model_PROPOSED(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [50, 0, 5, 0]
    ACT = ['linear', 'sigmoid', 'ReLU', 'TanH','Leaky ReLU']
    Optimizer = ['RMSprop', 'Adam', 'SRD', 'Adagrad', 'Adadelta']
    IMG_SIZE = [32, 32, 3]
    Feat = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    train_data = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Feat2[i, :] = cv.resize(test_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    test_data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Model.add(Attention())
    Model.add(Dense(5, activation=ACT[sol[3]]))
    model = dense_net(train_target.shape[1])
    model.compile(optimizer=Optimizer[sol[1]], loss='categorical_crossentropy', metrics=['accuracy'], Activation='Relu')
    model.fit(train_data, train_target, steps_per_epoch=sol[2], epochs=sol[0], batch_size=64)
    pred = model.predict(test_data)
    Eval = evaluation(pred, test_target)
    return Eval, pred
