import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from Image_Results import Image_Results
import matplotlib
from sklearn.metrics import roc_curve, roc_auc_score
from itertools import cycle


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def Plot_Results():
    matplotlib.use('TkAgg')
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Algorithm = ['TERMS', 'DHOA-DD-MHA', 'TSA-DD-MHA', 'JA-DD-MHA', 'GSO-DD-MHA', 'IRP-GSO-DD-MHA']
    Classifier = ['TERMS', 'VGG16', 'MobileNet', 'XGBoost', 'Densenet', 'IRP-GSO-DD-MHA']
    names = ['Dataset1 Epoch', 'Dataset2 Epoch']
    terms_used = [0, 1, 3, 8]
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('--------------------------------------------------', 'Dataset' + str(i + 1) + 'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)
        print()

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('--------------------------------------------------', 'Dataset' + str(i + 1) + 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)
        print()

    learnper = [45, 55, 65, 75, 85]
    for i in range(eval.shape[0]):
        for j in range(len(terms_used)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if terms_used[j] == 9:
                        Graph[k, l] = eval[i, k, l, terms_used[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, terms_used[j] + 4] * 100

            plt.plot(learnper, Graph[:, 0], color='#00FF00', linewidth=3, marker='*', markerfacecolor='red',
                     markersize=17,
                     label="DHOA-DD-MHA")
            plt.plot(learnper, Graph[:, 1], color='#FFFF00', linewidth=3, marker='*', markerfacecolor='green',
                     markersize=17,
                     label='TSA-DD-MHA')
            plt.plot(learnper, Graph[:, 2], color='#4169E1', linewidth=3, marker='*', markerfacecolor='magenta',
                     markersize=17,
                     label="JA-DD-MHA")
            plt.plot(learnper, Graph[:, 3], color='#FF69B4', linewidth=3, marker='*', markerfacecolor='blue',
                     markersize=17,
                     label="GSO-DD-MHA")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='black',
                     markersize=17,
                     label="IRP-GSO-DD-MHA")

            plt.xticks(learnper, ['100', '200', '300', '400', '500'])
            plt.xlabel('Epochs')
            plt.ylabel(Terms[terms_used[j]])
            plt.legend(loc=4)
            path1 = "./Results/%s_%s_line.png" % (names[i], Terms[terms_used[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='#00FF00', width=0.10, label="VGG16")
            ax.bar(X + 0.10, Graph[:, 6], color='m', width=0.10, label="MobileNet")
            ax.bar(X + 0.20, Graph[:, 7], color='#4169E1', width=0.10, label="XGBoost")
            ax.bar(X + 0.30, Graph[:, 8], color='#FFFF00', width=0.10, label="Densenet")
            ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="IRP-GSO-DD-MHA")
            plt.xticks(X + 0.10, ('100', '200', '300', '400', '500'))
            plt.ylabel(Terms[terms_used[j]])
            plt.xlabel('Epochs')
            plt.legend(loc=1)
            path1 = "./Results/%s_%s_bar.png" % (names[i], Terms[terms_used[j]])
            plt.savefig(path1)
            plt.show()


def Plot_segmet_Results():
    matplotlib.use('TkAgg')
    eval = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice', 'Jaccard']
    Classifier = ['TERMS', 'unet', 'transunet', 'without_optimization', 'proposed']
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, :]
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        Table.add_column(Classifier[1], value[1, :])
        Table.add_column(Classifier[2], value[2, :])
        Table.add_column(Classifier[3], value[3, :])
        Table.add_column(Classifier[4], value[4, :])

        print('--------------------------------------------------', 'Dataset' + str(i + 1) + '',
              '--------------------------------------------------')
        print(Table)
        print()


def Plot_ROC():
    lw = 2
    cls = ['VGG16', 'MobileNet', 'XGBoost', 'Densenet', 'IRP-GSO-DD-MHA']  # c
    for a in range(2):  # For 5 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')

        colors = cycle(["aqua", "#00FF00", "#FFFF00", "deeppink", "navy"])
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC__.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_results_conv():
    # matplotlib.use('TkAgg')
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'DHOA-DD-MHA', 'TSA-DD-MHA', 'JA-DD-MHA', 'GSO-DD-MHA', 'IRP-GSO-DD-MHA']

    for i in range(Result.shape[0]):
        length = np.arange(25)
        Conv_Graph = Fitness[i]
        # Conv_Graph = np.reshape(BestFit[i], (8, 20))
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
                 label='DHOA-DD-MHA')
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12,
                 label='TSA-DD-MHA')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12,
                 label='JA-DD-MHA')
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12,
                 label='GSO-DD-MHA')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12,
                 label='IRP-GSO-DD-MHA')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s_.png" % (i + 1))
        plt.show()


if __name__ == '__main__':
    Plot_Results()
    Plot_ROC()
    plot_results_conv()
    Image_Results()
    Plot_segmet_Results()
