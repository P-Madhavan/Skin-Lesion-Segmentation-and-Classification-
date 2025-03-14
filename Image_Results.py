import numpy as np
import matplotlib
import cv2 as cv
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

no_of_Datasets = 2
Datasets = ['HAM10000', 'PH2Dataset']


def Image_Results():
    for i in range(no_of_Datasets):
        Orig = np.load('Images_' + str(i + 1) + '.npy', allow_pickle=True)
        segment = np.load('segmentation_' + str(i + 1) + '.npy', allow_pickle=True)
        ground = np.load('GT_' + str(i + 1) + '.npy', allow_pickle=True)
        ind = [10, 20, 25, 30, 35]
        for j in range(len(ind)):
            original = Orig[ind[j]]
            seg = segment[ind[j]]
            GT = ground[ind[j]]
            fig, ax = plt.subplots(1, 3)
            plt.suptitle(Datasets[i], fontsize=20)
            plt.subplot(1, 3, 1)
            plt.title('Orig')
            plt.imshow(original)
            plt.subplot(1, 3, 2)
            plt.title('Seg')
            plt.imshow(seg)
            plt.subplot(1, 3, 3)
            plt.title('GT')
            plt.imshow(GT)
            path1 = "./Results/Image_Res/Dataset_%s_Image_%s_image.png" % (i + 1, j + 1)
            plt.savefig(path1)
            plt.show()
            cv.imwrite('./Results/Image_Res1/Dataset-' + str(i+1) + 'orig-' + str(j + 1) + '.png', original)
            cv.imwrite('./Results/Image_Res1/Dataset-' + str(i+1) + 'pre-proc-' + str(j + 1) + '.png', GT)
            cv.imwrite('./Results/Image_Res1/Dataset-' + str(i+1) + 'segment-' + str(j + 1) + '.png', seg)


if __name__ == '__main__':
    Image_Results()

