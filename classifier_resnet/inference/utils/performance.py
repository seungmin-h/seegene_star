#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : mathematics
# @Date : 2021-11-17-09-10
# @Project : seegene_challenge
# @Author : seungmin

from itertools import cycle
from sklearn.metrics import roc_curve, auc

import itertools
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from scipy import interp

def sspn(cm_value):
    #cm_value=confusion_matrix(np.argmax(label,1),np.argmax(pred,1))
    cm_sum=np.sum(cm_value)
    print(cm_sum)
    diag_sum=cm_value[0][0]+cm_value[1][1]+cm_value[1][1]+cm_value[2][2]
    print(diag_sum)
    print('Overall accuracy', str(round((diag_sum/cm_sum)*100., 3))+'%')
    print('')
    print('Normal class performance')
    TP=cm_value[0][0]
    FP=cm_value[1][0]+cm_value[2][0]
    FN=cm_value[0][1]+cm_value[0][2]
    TN=cm_sum-TP-FP-FN
    print('   ', 'TP:'+str(TP), 'TN:'+str(TN), 'FP:'+str(FP), 'FN:'+str(FN))
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    print('   ', 'Sensitivity:'+str(round(TPR*100, 3))+'%')
    print('   ', 'Specificity:'+str(round(TNR*100, 3))+'%')
    print('   ', 'PPV(precision):'+str(round(PPV*100, 3))+'%')
    print('   ', 'NPV:'+str(round(NPV*100, 3))+'%')
    print('')
    print('Dysplasia class performance')
    TP=cm_value[1][1]
    FP=cm_value[0][1]+cm_value[2][1]
    FN=cm_value[1][0]+cm_value[1][2]
    TN=cm_sum-TP-FP-FN
    print('   ', 'TP:'+str(TP), 'TN:'+str(TN), 'FP:'+str(FP), 'FN:'+str(FN))
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    print('   ', 'Sensitivity:'+str(round(TPR*100, 3))+'%')
    print('   ', 'Specificity:'+str(round(TNR*100, 3))+'%')
    print('   ', 'PPV(precision):'+str(round(PPV*100, 3))+'%')
    print('   ', 'NPV:'+str(round(NPV*100, 3))+'%')
    print('')
    print('Malignant class performance')
    TP=cm_value[2][2]
    FP=cm_value[0][2]+cm_value[1][2]
    FN=cm_value[2][0]+cm_value[2][1]
    TN=cm_sum-TP-FP-FN
    print('   ', 'TP:'+str(TP), 'TN:'+str(TN), 'FP:'+str(FP), 'FN:'+str(FN))
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    print('   ', 'Sensitivity:'+str(round(TPR*100, 3))+'%')
    print('   ', 'Specificity:'+str(round(TNR*100, 3))+'%')
    print('   ', 'PPV(precision):'+str(round(PPV*100, 3))+'%')
    print('   ', 'NPV:'+str(round(NPV*100, 3))+'%')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save='confusion_matrix.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    refence:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #plt.figure(figsize=(10,11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', labelpad=10)
    plt.xlabel('Predicted label', labelpad=0)
    plt.savefig(save)
    plt.close()

## roc curve, confusion matrix
def plt_roc(test_y, probas_y, plot_micro=True, plot_macro=True):
    assert isinstance(test_y, list) and isinstance(probas_y, list), 'the type of input must be list'
    skplt.metrics.plot_roc(test_y, probas_y, plot_micro=plot_micro, plot_macro=plot_macro, text_fontsize=15,
                           cmap=plt.cm.get_cmap('rainbow', 5),
                           figsize=(10,10))
    plt.savefig('roc_auc_curve.png')
    plt.close()


def plot_roc(true, prob, label):

    n_classes=len(label)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true[:, i], prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true.ravel(), prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    lw=2

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            
    # Plot all ROC curves
    plt.figure(figsize=(10,10))
    plt.plot(fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
            )

    plt.plot(fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
            )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        if i == 0:
            class_name='Normal'
        if i == 1:
            class_name='Dysplasia'
        if i == 2:
            class_name='Malignant'

        plt.plot(fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})".format(class_name, roc_auc[i]),
                )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=20)
        plt.ylabel("True Positive Rate", fontsize=20)
        #plt.title("Some extension of Receiver operating characteristic to multiclass")
        plt.legend(loc="lower right", fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig('roc_curve.png')
        plt.close()

