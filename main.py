import os

import keras.utils
from keras.layers import Reshape, BatchNormalization, Activation, MaxPooling2D, Conv2D, Dropout, GRU, Dense, \
    Input, Bidirectional, TimeDistributed, GlobalAveragePooling1D, Concatenate
from keras.models import Model


import Normalizer
import Metrics
import CallBacks

import random
import numpy.random as npr
from tensorflow import set_random_seed

from datasetGenerator import DCASE2018

if __name__ == '__main__':
    # deactivate warning (TMP)
    import warnings
    warnings.filterwarnings("ignore")

    # ARGUMENT PARSER ====
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer", help="normalizer [file_MinMax | global_MinMax | file_Mean | global_Mean | file_standard | global_standard | unit")
    parser.add_argument("--output_model", help="basename for save file of the model")
    args = parser.parse_args()

    # PROCESS ARGUMENTS ====
    normalizer = None
    if args.normalizer:
        if args.normalizer == "file_MinMax": normalizer = Normalizer.MinMaxScaler()
        if args.normalizer == "file_Mean": normalizer = Normalizer.MeanScaler()
        if args.normalizer == "file_Standard": normalizer = Normalizer.StandardScaler()
        if args.normalizer == "global_MinMax": normalizer = Normalizer.MinMaxScaler(methods="global")
        if args.normalizer == "global_Mean": normalizer = Normalizer.MeanScaler(methods="global")
        if args.normalizer == "global_Standard": normalizer = Normalizer.StandardScaler(methods="global")
        if args.normalizer == "unit": normalizer = Normalizer.UnitLength()

    # Prepare the save directory (if needed)
    dirPath = None
    if args.output_model is not None:
        dirPath = args.output_model
        dirName = os.path.dirname(dirPath)
        fileName = os.path.basename(dirPath)

        if not os.path.isdir(dirName):
            print("File doesn't exist, creating it")
            os.makedirs(dirName)

    # fix the random seeds
    seed = 1324
    random.seed(seed)
    npr.seed(seed)
    set_random_seed(seed)

    # GENERATE DATASET ====
    metaRoot = "meta"
    featRoot = "features_2"
    feat = ["mel", "stack"]
    dataset = DCASE2018(
        featureRoot=featRoot,
        metaRoot=metaRoot,
        features=feat,
        validationPercent=0.2,
        normalizer=normalizer
    )

    print(dataset)

    # MODEL HYPERPARAMETERS ====
    epochs = 100
    batch_size = 12
    metrics = ["binary_accuracy", Metrics.precision, Metrics.recall, Metrics.f1]
    loss = "binary_crossentropy"
    optimizer = "adam"
    callbacks = [
        CallBacks.CompleteLogger(logPath=dirPath, validation_data=([dataset.validationDataset["mel"]["input"],
                                                                    dataset.validationDataset["stack"]["input"]],
                                                                   dataset.validationDataset["mel"]["output"])
                                 ),
    ]

    # ==================================================================================================================
    #   Creating the model
    # ==================================================================================================================
    melInput = Input(dataset.getInputShape("mel"))
    stackInput = Input(dataset.getInputShape("stack"))

    # ---- mel convolution part ----
    mBlock1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(melInput)
    mBlock1 = BatchNormalization()(mBlock1)
    mBlock1 = Activation(activation="relu")(mBlock1)
    mBlock1 = MaxPooling2D(pool_size=(4, 1))(mBlock1)
    mBlock1 = Dropout(0.3)(mBlock1)

    mBlock2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(mBlock1)
    mBlock2 = BatchNormalization()(mBlock2)
    mBlock2 = Activation(activation="relu")(mBlock2)
    mBlock2 = MaxPooling2D(pool_size=(4, 1))(mBlock2)
    mBlock2 = Dropout(0.3)(mBlock2)

    mBlock3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(mBlock2)
    mBlock3 = BatchNormalization()(mBlock3)
    mBlock3 = Activation(activation="relu")(mBlock3)
    mBlock3 = MaxPooling2D(pool_size=(4, 1))(mBlock3)
    mBlock3 = Dropout(0.3)(mBlock3)

    targetShape = int(mBlock3.shape[1] * mBlock3.shape[2])
    mReshape = Reshape(target_shape=(targetShape, 64))(mBlock3)

    # ---- stack convolution part ----
    sBlock1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(stackInput)
    sBlock1 = BatchNormalization()(sBlock1)
    sBlock1 = Activation(activation="relu")(sBlock1)
    sBlock1 = MaxPooling2D(pool_size=(1, 2))(sBlock1)
    sBlock1 = Dropout(0.3)(sBlock1)

    sBlock2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(sBlock1)
    sBlock2 = BatchNormalization()(sBlock2)
    sBlock2 = Activation(activation="relu")(sBlock2)
    sBlock2 = MaxPooling2D(pool_size=(1, 2))(sBlock2)
    sBlock2 = Dropout(0.3)(sBlock2)

    sBlock3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(sBlock2)
    sBlock3 = BatchNormalization()(sBlock3)
    sBlock3 = Activation(activation="relu")(sBlock3)
    sBlock3 = MaxPooling2D(pool_size=(1, 2))(sBlock3)
    sBlock3 = Dropout(0.3)(sBlock3)

    targetShape = int(sBlock3.shape[1] * sBlock3.shape[2])
    sReshape = Reshape(target_shape=(targetShape, 64))(sBlock3)

    # ---- concatenate ----
    conc = Concatenate(axis=1)([mReshape, sReshape])

    gru = Bidirectional(
        GRU(kernel_initializer='glorot_uniform', recurrent_dropout=0.0, dropout=0.3, units=64, return_sequences=True)
    )(conc)
    print(gru.shape)

    output = TimeDistributed(
        Dense(dataset.nbClass, activation="sigmoid"),
    )(gru)
    print(output.shape)

    output = GlobalAveragePooling1D()(output)

    model = Model(inputs=[melInput, stackInput], outputs=output)
    keras.utils.print_summary(model, line_length=100)

    # compile & fit model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(
        x=[dataset.trainingDataset["mel"]["input"],dataset.trainingDataset["stack"]["input"]],
        y=dataset.trainingDataset["mel"]["output"],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            [dataset.validationDataset["mel"]["input"], dataset.validationDataset["stack"]["input"]],
            dataset.validationDataset["mel"]["output"]
        ),
        callbacks=callbacks,
        verbose=0
    )

    # save json
    model_json = model.to_json()
    with open(dirPath + "_model.json", "w") as f:
        f.write(model_json)

    # save weight
    model.save_weights(dirPath + "_weight.h5py")


