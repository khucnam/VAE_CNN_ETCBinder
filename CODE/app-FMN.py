from model.vae import VAE
from utility.io import load_data
import pandas as pd
import keras
from matplotlib import pyplot
# %matplotlib inline
from keras.models import Sequential
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Conv1D, MaxPooling1D, \
    AveragePooling1D, UpSampling1D
from keras.utils import np_utils
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_curve, auc
import os, math
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, roc_auc_score, accuracy_score
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.utils import class_weight
import tensorflow
from numpy.random import seed

tensorflow.random.set_seed(2)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import numpy as np
from keras.regularizers import l2, l1
from keras.callbacks import LearningRateScheduler
from keras import initializers

initializer = tf.keras.initializers.GlorotNormal()


def labelToOneHot(label):  # 0--> [1 0], 1 --> [0 1]
    label = label.reshape(len(label), 1)
    label = np.append(label, label, axis=1)
    label[:, 0] = label[:, 0] == 0;
    return label


def multi_fold_data(file_path):
    dataset = pd.read_csv(file_path, header=None)
    row, col = dataset.shape[0], dataset.shape[1]
    numOfPos = len(dataset[dataset[col - 1] == 1])
    print("numOfPos ", numOfPos, "numOfNeg ", row - numOfPos)
    time = (row - numOfPos) // numOfPos
    extra = row - time * numOfPos
    pos_data = dataset[dataset[col - 1] == 1]
    multi_fold = pos_data
    for i in range(time - 3):
        multi_fold = pd.concat([multi_fold, pos_data])

    multi_fold = pd.concat([multi_fold, pos_data.sample(n=extra, replace=True)])
    print("multi_fold type ", type(multi_fold))
    print("multi_fold shape ", multi_fold.shape)
    return multi_fold


def CNN(generatedBalancedData, CNN_dropout, CNN_re):
    # define model
    input_sig = Input(shape=(generatedBalancedData.shape[1] * generatedBalancedData.shape[2], 1))
    x1 = Conv1D(64, 32, activation='relu', padding='same', kernel_initializer=initializer)(input_sig)
    x2 = MaxPooling1D(2)(x1)
    x3 = Conv1D(128, 32, activation='relu', padding='same', kernel_initializer=initializer)(x2)
    x4 = Dropout(CNN_dropout)(x3)
    x5 = MaxPooling1D(2)(x4)
    x6 = Conv1D(256, 32, activation='relu', padding='same', kernel_initializer=initializer)(x5)
    x7 = Dropout(CNN_dropout)(x6)
    x8 = MaxPooling1D(2)(x7)
    x9 = Dropout(CNN_dropout)(x8)
    flat = Flatten()(x9)
    d1 = Dense(256, activation='relu', kernel_regularizer=l2(CNN_re), kernel_initializer=initializer)(flat)
    d2 = Dense(512, activation='relu', kernel_regularizer=l2(CNN_re), kernel_initializer=initializer)(d1)
    output_layer = Dense(2, activation='softmax')(d2)

    cnn = Model(input_sig, output_layer)
    cnn.summary()
    return cnn


def classificationPerformanceByThreshold(threshold, y_pred, y_test):
    auc = roc_auc_score(y_test, y_pred)
    y_test = np.argmax(y_test, axis=1)

    Y_pred = np.empty_like(y_pred)
    for i in range(len(y_pred)):
        if y_pred[i][0] >= threshold:
            Y_pred[i] = np.array([1, 0])  # assign as class pos
        else:
            Y_pred[i] = np.array([0, 1])  # assign as class neg

    Y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(y_test, Y_pred, labels=[0, 1])

    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]

    if float(tp) + float(fn) == 0:
        TPR = round(float(tp) / 0.00000001, 3)
    else:
        TPR = round(float(tp) / (float(tp) + float(fn)), 3)

    if float(fp) + float(tn) == 0:
        FPR = round(float(fp) / (0.00000001), 3)
    else:
        FPR = round(float(fp) / (float(fp) + float(tn)), 3)

    if float(tp) + float(fp) + float(fn) + float(tn) == 0:
        accuracy = round((float(tp) + float(tn)) / (0.00000001), 3)
    else:
        accuracy = round((float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn)), 3)

    if float(tn) + float(fp) == 0:
        specitivity = round(float(tn) / (0.00000001), 3)
    else:
        specitivity = round(float(tn) / (float(tn) + float(fp)), 3)

    if float(tp) + float(fn) == 0:
        sensitivity = round(float(tp) / (0.00000001), 3)
    else:
        sensitivity = round(float(tp) / (float(tp) + float(fn)), 3)

    if float(tp) + float(fp) == 0:
        precision = round(float(tp) / (0.00000001), 3)
    else:
        precision = round(float(tp) / (float(tp) + float(fp)), 3)

    if math.sqrt(
            (float(tp) + float(fp)) * (float(tp) + float(fn)) * (float(tn) + float(fp)) * (float(tn) + float(fn))) == 0:
        mcc = round((float(tp) * float(tn) - float(fp) * float(fn)) / 0.00000001, 3)
    else:
        mcc = round((float(tp) * float(tn) - float(fp) * float(fn)) / math.sqrt(
            (float(tp) + float(fp))
            * (float(tp) + float(fn))
            * (float(tn) + float(fp))
            * (float(tn) + float(fn))
        ), 3)
    balAcc = (sensitivity + specitivity) / 2
    if (sensitivity + precision) == 0:
        f_measure = round(2 * sensitivity * precision / (0.00000001), 3)
    else:
        f_measure = round(2 * sensitivity * precision / (sensitivity + precision), 3)

    return accuracy, specitivity, sensitivity, mcc, tp, tn, fp, fn, TPR, FPR, balAcc, precision, f_measure, auc


def trainAndPredict(latent_dim, k, pathToSaveModel, folderForImage, trainFile, testFile, VAE_epochs, VAE_batch_size,
                    VAE_lr, CNN_epochs, CNN_batch_size, CNN_lr, CNN_dropout, CNN_re, wd, fold, fullRSFile, PRfile,
                    fileForEpochPlotting, fileForCurvePlotting):
    # train input
    dataset = pd.read_csv(trainFile, header=None)
    X_train = dataset.iloc[:, 0:wd * 20].values
    X_train = X_train.reshape(X_train.shape[0], wd * 20, 1)
    X_train = X_train.astype(float)
    # print("X_train ", X_train[:10])
    y_train = dataset.iloc[:, wd * 20].values
    print(y_train.shape)
    y_train = labelToOneHot(y_train)
    print(y_train.shape)

    # train input for pos only
    X_train_pos = dataset[dataset[dataset.shape[1] - 1] == 1]
    X_train_pos = X_train_pos.iloc[:, 0:wd * 20].values
    X_train_pos = X_train_pos.reshape(X_train_pos.shape[0], wd * 20, 1)
    X_train_pos = X_train_pos.astype(float)

    # test input
    dataset = pd.read_csv(testFile, header=None)
    X_test = dataset.iloc[:, 0:wd * 20].values
    X_test = X_test.reshape(X_test.shape[0], wd * 20, 1)
    X_test = X_test.astype(float)
    y_test = dataset.iloc[:, wd * 20].values
    y_test = labelToOneHot(y_test)

    vae = VAE(latent_dim, k, VAE_lr, VAE_epochs, VAE_batch_size)
    # Train VAE
    vae.fit(X_train_pos)
    # input to generate latent sample
    multi_fold = multi_fold_data(trainFile)
    multi_fold = multi_fold.iloc[:, 0:wd * 20].values
    multi_fold = multi_fold.reshape(multi_fold.shape[0], wd * 20, 1)
    multi_fold = multi_fold.astype(float)

    # Sampling 1 data point from each data point in train set
    generated_data = vae.generate(X=multi_fold)

    print("generated_data shape ", generated_data.shape)
    print("X_train_pos shape ", X_train_pos.shape)
    print("X_train shape ", X_train.shape)
    print("generated_data type ", type(generated_data))
    print("X_train_pos type ", type(X_train_pos))
    print("X_train type ", type(X_train))

    X_train_balance = np.vstack((generated_data, X_train))
    print(" X_train_balance shape ", X_train_balance.shape)

    y_train_generated = np.ones((len(generated_data),), dtype=int)
    print("y_train_generated ", y_train_generated.shape)
    y_train_generated = labelToOneHot(y_train_generated)
    y_train_balance = np.vstack((y_train_generated, y_train))
    print("y_train_balance ", y_train_balance.shape)

    cnnModel = CNN(X_train_balance, CNN_dropout, CNN_re)
    adam = Adam(lr=CNN_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    cnnModel.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30)
    cp = ModelCheckpoint(pathToSaveModel + "/fold" + str(fold) + "_epoch" + str(CNN_epochs) + ".batch_size" + str(
        CNN_batch_size) + ".lr" + str(CNN_lr) + ".CNN_dropout" + str(CNN_dropout) + "CNN_re" + str(CNN_re) + ".CNN.h5",
                         monitor='loss', mode='min', verbose=2, save_best_only=True)
    CNN_history = cnnModel.fit(X_train_balance, y_train_balance, batch_size=CNN_batch_size, epochs=CNN_epochs,
                               verbose=1, callbacks=[es])

    with open(fileForEpochPlotting, mode='w') as f:
        f.write("epoch,accuracy,loss\n")
        epoch = 0
        for i, j in zip(CNN_history.history['accuracy'], CNN_history.history['loss']):
            f.write(str(epoch) + "," + str(i) + "," + str(j) + "\n")
            epoch += 1
    y_pred = cnnModel.predict(X_test)

    f2 = open(fullRSFile, "a")
    threshold = 0.001
    while threshold < 1:
        accuracy, specitivity, sensitivity, mcc, tp, tn, fp, fn, TPR, FPR, balAcc, precision, f_measure, auc_value = classificationPerformanceByThreshold(
            threshold, y_pred, y_test)
        threshold = threshold + 0.002
        f2.write(str(threshold) + ", " + str(accuracy) + ", " + str(specitivity) + ", " + str(sensitivity) + ", " + str(
            mcc) + ", " + str(precision) + ", " + str(tp) + ", " + str(tn) + ", " + str(fp) + ", " + str(
            fn) + ", " + str(TPR) + ", " + str(FPR) + ", " + str(balAcc) + ", " + str(precision) + ", " + str(
            f_measure) + ", " + str(auc_value) + "\n")
    f2.close()

    # calculate roc curves
    # fpr, tpr, _ = roc_curve(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred[:, 1])
    # plot the roc curve for the model
    roc_auc = roc_auc_score(y_test, y_pred)
    # pyplot.plot(fpr, tpr, marker='.', label="CNN-AU ROC is {0:.3f}%".format(roc_auc) )
    # # axis labels
    # pyplot.xlabel('False Positive Rate')
    # pyplot.ylabel('True Positive Rate')
    # # show the legend
    # pyplot.legend()
    # pyplot.title(str(fold)+"_epoch"+str(CNN_epochs)+".batch_size"+str(CNN_batch_size)+".lr"+str(CNN_lr)+".CNN_dropout"+str(CNN_dropout)+"CNN_re"+str(CNN_re))
    # #save the plot

    # pyplot.savefig(folderForImage+"/"+str(fold)+"_epoch"+str(CNN_epochs)+".batch_size"+str(CNN_batch_size)+".lr"+str(CNN_lr)+".CNN_dropout"+str(CNN_dropout)+"CNN_re"+str(CNN_re)+".AUROC.png")
    # # show the plot
    # pyplot.show()

    # convert the prediction to a pandas DataFrame:
    ypred_df = pd.DataFrame(y_pred)
    # save to csv:
    with open(fileForCurvePlotting, mode='w') as f:
        ypred_df.to_csv(f)

    pos_probs = y_pred[:, 1]
    # calculate PR roc curve for model
    # precisions, recalls, thresholds  = precision_recall_curve(y_test, pos_probs)
    # precisions, recalls, thresholds  = precision_recall_curve(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    precisions, recalls, thresholds = precision_recall_curve(y_test[:, 1], y_pred[:, 1])
    # calculate the precision-recall auc
    auc_score = auc(recalls, precisions)
    # plot
    # pyplot.plot(recalls, precisions, marker='.', label="CNN-AUPR curve is {0:.3f}%".format(auc_score))
    # # axis labels
    # pyplot.xlabel('Recall')
    # pyplot.ylabel('Precision')
    # # show the legend
    # pyplot.legend()
    # pyplot.title(str(fold)+"_epoch"+str(CNN_epochs)+".batch_size"+str(CNN_batch_size)+".lr"+str(CNN_lr)+".CNN_dropout"+str(CNN_dropout)+"CNN_re"+str(CNN_re))
    # #save to image
    # pyplot.savefig(folderForImage+"/"+str(fold)+"_epoch"+str(CNN_epochs)+".batch_size"+str(CNN_batch_size)+".lr"+str(CNN_lr)+".CNN_dropout"+str(CNN_dropout)+"CNN_re"+str(CNN_re)+".AUPR.png")
    # # show the plot
    # pyplot.show()

    f = open(PRfile, "w")
    f.write("threshold,recall,precision,PRauc_score,roc_auc\n")
    # print("type(y_test) ",type(y_test), " ",y_test.shape," y_test.argmax(axis=1) ",y_test.argmax(axis=1).shape)
    # print("type(y_pred) ",type(y_pred), " ",y_pred.shape)
    # print("type(thresholds) ",type(thresholds), " ",thresholds.shape)
    # print("type(precisions) ",type(precisions), " ",precisions.shape)
    # print("type(recalls) ",type(recalls), " ",recalls.shape)
    for threshold, precision, recall in zip(thresholds, precisions[:-1], recalls[:-1]):
        f.write(str(threshold) + "," + str(recall) + "," + str(precision) + "," + str(auc_score) + "," + str(
            roc_auc) + "\n")
    f.close()
    return roc_auc, auc_score


def main():
    bindingTypes = ["FMN"]
    wd = 15
    for bdType in bindingTypes:
        folder = "FMN/pssm features wd 15"
        for VAE_epoch in [5]:
            for VAE_batch_size in [64]:
                for VAE_lr in [0.0001]:
                    for CNN_dropout in [0.5]:
                        for CNN_epoch in [3]:
                            for CNN_batch_size in [128]:
                                for CNN_re in [1e-4]:
                                    for CNN_lr in [0.0001]:
                                        for latentDim in [64]:
                                            for k in [0.4]:
                                                for time in range(1):
                                                    pathToSaveModel = "FMN/pssm features wd 15/dim" + str(
                                                        latentDim) + "/New_VAE_CNN results/Saved models"
                                                    folderForImage = "FMN/pssm features wd 15/dim" + str(
                                                        latentDim) + "/New_VAE_CNN results"
                                                    f = open(folder + "/New_VAE_CNN results/dim" + str(
                                                        latentDim) + "/VAE_CNN_arch1.Overall.Results.csv", "a")
                                                    #                                      set_rd_seed()
                                                    inputFileTrain = folder + "/input.train.csv"
                                                    inputFileTest = folder + "/ind.test.csv"
                                                    rsFile = folder + "/New_VAE_CNN results/dim" + str(
                                                        latentDim) + "/time" + str(time) + "k" + str(
                                                        k) + ".epoch" + str(CNN_epoch) + "_batchsize" + str(
                                                        CNN_batch_size) + "_lr" + str(CNN_lr) + ".CNN_dropout" + str(
                                                        CNN_dropout) + "CNN_re" + str(CNN_re) + "_ind.result.csv"
                                                    fullRSFile = folder + "/New_VAE_CNN results/dim" + str(
                                                        latentDim) + "/FullRS.k" + str(k) + ".time" + str(
                                                        time) + ".epoch" + str(CNN_epoch) + "_batchsize" + str(
                                                        CNN_batch_size) + "_lr" + str(CNN_lr) + ".CNN_dropout" + str(
                                                        CNN_dropout) + "CNN_re" + str(CNN_re) + "_ind.result.csv"
                                                    roc_auc, auc_score = trainAndPredict(latentDim, k, pathToSaveModel,
                                                                                         folderForImage, inputFileTrain,
                                                                                         inputFileTest, VAE_epoch,
                                                                                         VAE_batch_size, VAE_lr,
                                                                                         CNN_epoch, CNN_batch_size,
                                                                                         CNN_lr, CNN_dropout, CNN_re,
                                                                                         wd, 0, fullRSFile,
                                                                                         rsFile[:-4] + ".PR.csv",
                                                                                         rsFile[:-4] + ".epochPlot.csv",
                                                                                         rsFile[:-4] + "yPred.csv")
                                                    f.write("k" + str(k) + "time" + str(time) + ".latentDim" + str(
                                                        latentDim) + ".epoch" + str(CNN_epoch) + "_batchsize" + str(
                                                        CNN_batch_size) + "_lr" + str(CNN_lr) + ".CNN_dropout" + str(
                                                        CNN_dropout) + ".CNN_re" + str(CNN_re) + "_ind,")
                                                    f.write(str(roc_auc) + "," + str(auc_score) + "\n")

                                                    # for fold in [1,2,3,4,5]:
                                                    # set_rd_seed()
                                                    # inputFileTrain=folder+"/input.fold.train"+str(fold)+".csv"
                                                    # inputFileTest=folder+"/input.fold.test"+str(fold)+".csv"
                                                    # rsFile=folder+"/VAE_CNN results/compareDifferentLatentDimArch1/pos.gen.time"+str(time)+".epoch"+str(CNN_epoch)+"_batchsize"+str(CNN_batch_size)+"_lr"+str(CNN_lr)+".CNN_dropout"+str(CNN_dropout)+"CNN_re"+str(CNN_re)+"_fold.result"+str(fold)+".csv"
                                                    # fullRSFile=folder+"/VAE_CNN results/compareDifferentLatentDimArch1/FullRS.pos.gen.time"+str(time)+".epoch"+str(CNN_epoch)+"_batchsize"+str(CNN_batch_size)+"_lr"+str(CNN_lr)+".CNN_dropout"+str(CNN_dropout)+"CNN_re"+str(CNN_re)+"_fold.result"+str(fold)+".csv"
                                                    # roc_auc, auc_score=run(latentDim, pathToSaveModel, folderForImage, inputFileTrain, inputFileTest,  VAE_epoch, VAE_batch_size, VAE_lr, CNN_epoch, CNN_batch_size, CNN_lr, CNN_dropout, CNN_re, wd, fold,fullRSFile, rsFile[:-4]+".PR.csv",rsFile[:-4]+".epochPlot.csv",rsFile[:-4]+"yPred.csv")
                                                    # f.write("time"+str(time)+".latentDim"+str(latentDim)+".epoch"+str(CNN_epoch)+"_batchsize"+str(CNN_batch_size)+"_lr"+str(CNN_lr)+".CNN_dropout"+str(CNN_dropout)+"CNN_re"+str(CNN_re)+".fold"+str(fold)+",")
                                                    # f.write(str(roc_auc)+","+str(auc_score)+"\n")
                                                    f.close()


if __name__ == "__main__":
    main()
