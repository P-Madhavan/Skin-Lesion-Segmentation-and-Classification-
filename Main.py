import os
import cv2 as cv
import numpy as np
import pandas as pd
from numpy import matlib
from DHOA import DHOA
from GSO import GSO
from Global_Vars import Global_vars
from Image_Results import Image_Results
from JAYA import JAYA
from Model_MobileNet import Model_MobileNet
from Model_PROPOSED import Model_PROPOSED
from Model_Transunet import Model_TransUnet
from Model_VGG16 import Model_VGG16
from Model_XGBOOST import Model_XGBOOST
from Obj_Cls import Obj_fun, Objective_Seg
from PROPOSED import PROPOSED
from Plot_Results import *
from TSA import TSA

no_of_dataset = 2


def ReadText(filename):
    f = open(filename, "r")
    lines = f.readlines()
    Tar = []
    fileNames = []
    for lineIndex in range(len(lines)):
        if lineIndex and '||' in lines[lineIndex]:
            line = [i.strip() for i in lines[lineIndex].strip().strip('||').replace('||', '|').split('|')]
            fileNames.append(line[0])
            Tar.append(int(line[2]))
    Tar = np.asarray(Tar)
    uniq = np.unique(Tar)
    Target = np.zeros((len(Tar), len(uniq))).astype('int')
    for i in range(len(uniq)):
        index = np.where(Tar == uniq[i])
        Target[index, i] = 1
    return fileNames, Target


def Read_Image(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (512, 512))
    return image


def Read_Images(Directory):
    Images = []
    out_folder = os.listdir(Directory)
    for i in range(len(out_folder)):
        print(i)
        filename = Directory + out_folder[i]
        image = Read_Image(filename)
        Images.append(image)
    return Images


def Read_Datset_PH2(Directory, fileNames):
    Images = []
    GT = []
    folders = os.listdir(Directory)
    for i in range(len(folders)):
        if folders[i] in fileNames:
            image = Read_Image(Directory + folders[i] + '/' + folders[i] + '_Dermoscopic_Image/' + folders[i] + '.bmp')
            gt = Read_Image(Directory + folders[i] + '/' + folders[i] + '_lesion/' + folders[i] + '_lesion.bmp')
            Images.append(image)
            GT.append(gt)
    return Images, GT


def Read_CSV(Path):
    df = pd.read_csv(Path)
    values = df.to_numpy()
    value = values[:, 6]
    uniq = np.unique(value)
    Target = np.zeros((len(value), len(uniq))).astype('int')
    for i in range(len(uniq)):
        index = np.where(value == uniq[i])
        Target[index, i] = 1
    return Target


# Read Datasets
an = 0
if an == 1:
    Images1 = Read_Images('./Datasets/HAM10000/Images/')
    np.save('Images_1.npy', Images1)

    Target1 = Read_CSV('./Datasets/HAM10000/HAM10000_metadata.csv')
    np.save('Target_1.npy', Target1)

    fileNames, Target2 = ReadText('./Datasets/PH2Dataset/PH2_dataset.txt')
    Images2, GT = Read_Datset_PH2('./Datasets/PH2Dataset/PH2 Dataset images/', fileNames)
    np.save('Images_2.npy', Images2)
    np.save('GT_2.npy', GT)
    np.save('Target_2.npy', Target2)

# GroundTruth for Dataset1
an = 0
if an == 1:
    im = []
    img = np.load('Images_1.npy', allow_pickle=True)
    for i in range(len(img)):
        print(i)
        image = img[i]
        minimum = int(np.min(image))
        maximum = int(np.max(image))
        Sum = ((minimum + maximum) / 2)
        ret, thresh = cv.threshold(image, Sum, 255, cv.THRESH_BINARY_INV)
        im.append(thresh)
    np.save('GT_1.npy', im)

# pre-processing #
an = 0
if an == 1:
    for i in range(no_of_dataset):
        Img = np.load('Images_' + str(i + 1) + '.npy', allow_pickle=True)
        Preprocess = []
        for j in range(len(Img)):
            imges = Img[j]
            Orig = cv.resize(imges, (228, 228))
            # applying the median filter
            img = cv.medianBlur(Orig, 3)
            # The declaration of CLAHE
            # clipLimit -> Threshold for contrast limiting
            clahe = cv.createCLAHE(clipLimit=2)
            img = clahe.apply(img)
            Preprocess.append(img)
        np.save('Preprocess_' + str(i + 1) + '.npy', Preprocess)

##Optimization for Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat = np.load('Preprocess_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_vars.Feat = Feat
        Global_vars.Target = Target
        Npop = 10
        Ch_len = 4
        xmin = matlib.repmat(([4, 5, 0.01, 0.01]), Npop, 1)
        xmax = matlib.repmat(([64, 15, 0.99, 0.05]), Npop, 1)
        initsol = np.zeros((xmax.shape))
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Objective_Seg
        Max_iter = 50

        print("DHOA...")
        [bestfit1, fitness1, bestsol1, time1] = DHOA(initsol, fname, xmin, xmax, Max_iter)

        print("TSA...")
        [bestfit2, fitness2, bestsol2, time2] = TSA(initsol, fname, xmin, xmax, Max_iter)

        print("JAYA...")
        [bestfit3, fitness3, bestsol3, time3] = JAYA(initsol, fname, xmin, xmax, Max_iter)

        print("GSO...")
        [bestfit4, fitness4, bestsol4, time4] = GSO(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        best = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])

        np.save('Best_Sol_Opt' + str(n + 1) + '.npy', best)

##Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        seg = []
        sol = np.load('Best_Sol_Opt' + str(n + 1) + '.npy', allow_pickle=True)
        image = np.load('Preprocess_' + str(n + 1) + '.npy', allow_pickle=True)
        for i in range(len(image)):
            Img = image[i]
            best_sol = sol.astype('int')
            Segmet = Model_TransUnet(Img, best_sol)
            seg.append(Segmet)
            np.save('segmentation_' + str(n + 1) + '.npy', seg)

### OPTIMIZATION
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat = np.load('Preprocess_' + str(n + 1) + '.npy', allow_pickle=True)
        Tar = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_vars.Feat = Feat
        Global_vars.Target = Tar
        Npop = 10
        Chlen = 4
        xmin = np.matlib.repmat(([50, 0, 5, 0]), Npop, 1)
        xmax = np.matlib.repmat(([100, 4, 50, 4]), Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(Chlen):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Obj_fun
        max_iter = 50

        print('DHOA....')
        [bestfit1, fitness1, bestsol1, Time1] = DHOA(initsol, fname, xmin, xmax, max_iter)

        print('TSA....')
        [bestfit2, fitness2, bestsol2, Time2] = TSA(initsol, fname, xmin, xmax, max_iter)

        print('JAYA....')
        [bestfit3, fitness3, bestsol3, Time3] = JAYA(initsol, fname, xmin, xmax, max_iter)

        print('GSO....')
        [bestfit4, fitness4, bestsol4, Time4] = GSO(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

        Sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_CLS_' + str(n + 1) + '.npy', Sol)

### Classification
an = 0
if an == 1:
    Eval_all = []
    for n in range(no_of_dataset):
        Feature = np.load('Preprocess_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_CLS_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat = Feature
        EVAL = []
        Epoch = ['100', '200', '300', '400', '500']
        for learn in range(len(Epoch)):
            Epochs = round(Feat.shape[0] * 0.75)
            Train_Data = Feat[:Epochs, :]
            Train_Target = Target[:Epochs, :]
            Test_Data = Feat[Epochs:, :]
            Test_Target = Target[Epochs:, :]
            Eval = np.zeros((10, 14))
            for j in range(BestSol.shape[0]):
                print(learn, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target,
                                                  sol[j].astype('int'))
            Eval[5, :], pred1 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred2 = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred3 = Model_XGBOOST(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred4 = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :], pred5 = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Eval_all.npy', Eval_all)

Plot_Results()
Plot_ROC()
plot_results_conv()
Image_Results()
Plot_segmet_Results()
