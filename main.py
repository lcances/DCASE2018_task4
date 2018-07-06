import os
import argparse

import keras.utils
from keras.layers import Reshape, BatchNormalization, Activation, MaxPooling2D, Conv2D, Dropout, GRU, Dense
from keras.layers import Input, Bidirectional, TimeDistributed, GlobalAveragePooling1D, Concatenate
from keras.models import Model

import random
import numpy.random as npr
from tensorflow import set_random_seed

import Normalizer
import Metrics
import CallBacks
from datasetGenerator import DCASE2018

class CustomGRU(GRU):

    def get_config(self):
        print("getting configuration")
        config = super().get_config()
        config["temporal_weight"] = self.temporal_weight
        print(config)
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)


if __name__ == '__main__':
    # ==================================================================================================================
    #       MANAGE PROGRAM ARGUMENTS
    # ==================================================================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer", help="normalizer [file_MinMax | global_MinMax | file_Mean | global_Mean | file_standard | global_standard | unit")
    parser.add_argument("--output_model", help="basename for save file of the model")
    parser.add_argument("-w", help="If set, display the warnigs", action="store_true")
    args = parser.parse_args()

    normalizer = None
    if args.normalizer:
        if args.normalizer == "file_MinMax": normalizer = Normalizer.MinMaxScaler()
        if args.normalizer == "file_Mean": normalizer = Normalizer.MeanScaler()
        if args.normalizer == "file_Standard": normalizer = Normalizer.StandardScaler()
        if args.normalizer == "global_MinMax": normalizer = Normalizer.MinMaxScaler(methods="global")
        if args.normalizer == "global_Mean": normalizer = Normalizer.MeanScaler(methods="global")
        if args.normalizer == "global_Standard": normalizer = Normalizer.StandardScaler(methods="global")
        if args.normalizer == "unit": normalizer = Normalizer.UnitLength()

    if not args.w:
        import warnings
        warnings.filterwarnings("ignore")


    # ==================================================================================================================
    #       INITIALIZE THE PROGRAM AND PREPARE SAVE DIRECTORIES
    # ==================================================================================================================
    # prepare directory
    dirPath = None
    if args.output_model is not None:
        dirPath = args.output_model
        dirName = os.path.dirname(dirPath)
        fileName = os.path.basename(dirPath)

        if not os.path.isdir(dirName):
            os.makedirs(dirName)

    # fix the random seeds
    seed = 1324
    random.seed(seed)
    npr.seed(seed)
    set_random_seed(seed)

    # ==================================================================================================================
    #       PREPARE DATASET AND SETUP MODEL HYPER PARAMETERS
    # ==================================================================================================================
    # prepare dataset
    metaRoot = "../Corpus/DCASE2018/meta"
    featRoot = "../Corpus/DCASE2018/features_2"
    feat = ["mel"]
    dataset = DCASE2018(
        featureRoot=featRoot,
        metaRoot=metaRoot,
        features=feat,
        validationPercent=0.2,
        normalizer=normalizer
    )

    # hyperparameters
    epochs = 100
    batch_size = 12
    metrics = ["binary_accuracy", Metrics.precision, Metrics.recall, Metrics.f1]
    loss = "binary_crossentropy"
    optimizer = "adam"
    callbacks = [
        CallBacks.CompleteLogger(logPath=dirPath, validation_data=(dataset.validationDataset["mel"]["input"],
                                                                   dataset.validationDataset["mel"]["output"])
                                 ),
    ]

    # ==================================================================================================================
    #   Creating the model
    # ==================================================================================================================
    melInput = Input(dataset.getInputShape("mel"))

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

    gru = Bidirectional(
        GRU(kernel_initializer='glorot_uniform', recurrent_dropout=0.0, dropout=0.3, units=64, return_sequences=True)
    )(mReshape)

    output = TimeDistributed(
        Dense(dataset.nbClass, activation="sigmoid"),
    )(gru)

    output = GlobalAveragePooling1D()(output)

    model = Model(inputs=[melInput], outputs=output)
    keras.utils.print_summary(model, line_length=100)

    # compile & fit model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(
        x=dataset.trainingDataset["mel"]["input"],
        y=dataset.trainingDataset["mel"]["output"],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            dataset.validationDataset["mel"]["input"],
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


