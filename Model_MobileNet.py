import cv2 as cv
import numpy as np
from PIL import Image
from keras.applications import MobileNet
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from Evaluation import evaluation


def Model_MobileNet(Train_Data, Train_Target, test_data, test_tar):
    model = MobileNet(weights='imagenet')
    IMG_SIZE = [224, 224, 3]
    Feat = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Feat[i, :] = cv.resize(Train_Data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Train_Data = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    for i in range(Train_Data.shape[0]):
        data = Image.fromarray(np.uint8(Train_Data[i])).convert('RGB')
        data = image.img_to_array(data)
        data = np.expand_dims(data, axis=0)
        data = np.squeeze(data)
        Train_Data[i] = cv.resize(data, (224, 224))
        Train_Data[i] = preprocess_input(Train_Data[i])
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])  # optimizer='Adam'
    model.fit_generator(Train_Data, Train_Target,
                        steps_per_epoch=100,
                        epochs=10)
    pred = model.predict(test_data)
    Eval = evaluation(pred, test_tar)
    return Eval, pred
