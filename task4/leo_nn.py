import os
import numpy as np
from random import shuffle

import keras.utils
from keras.models import Model
from keras.layers import Reshape, BatchNormalization, Activation, MaxPooling2D, Conv2D, Dropout, GRU, Dense,\
    Input, Flatten, Bidirectional, TimeDistributed, GlobalAveragePooling1D

class DCASE2018:
    NB_CLASS = 10
    class_correspondance = {
        "Alarm_bell_ringing": 0,
        "Speech" : 1,
        "Dog": 2,
        "Cat": 3,
        "Vacuum_cleaner": 4,
        "Dishes": 5,
        "Frying": 6,
        "Electric_shaver_toothbrush": 7,
        "Blender": 8,
        "Running_water": 9}

    def __init__(self,
                 feat_train_weak:str = "", feat_train_unlabelInDomain: str = "", feat_train_unlabelOutDomain: str = "",
                 meta_train_weak: str = "", meta_train_unlabelInDomain: str = "", meta_train_unlabelOutDomain: str = "",
                 feat_test: str = "", meta_test: str = "",  validationPercent: float = 0.2, nbClass: int = 10,
                 nb_sequence: int = 1):

        #directories
        self.feat_train_weak = feat_train_weak
        self.feat_train_uid = feat_train_unlabelInDomain
        self.feat_train_uod = feat_train_unlabelOutDomain
        self.feat_test = feat_test
        self.meta_train_weak = meta_train_weak
        self.meta_train_uid = meta_train_unlabelInDomain
        self.meta_train_uod = meta_train_unlabelOutDomain
        self.meta_test = meta_test

        # dataset parameters
        self.metadata = {
            "weak": [],
            "uid": [],
            "uod": [],
            "test": []
        }

        self.validationPercent = validationPercent
        self.nbClass = nbClass
        self.originalShape = None

        # dataset that will be used
        self.nbSequence = nb_sequence
        self.trainingDataset = {"input": [], "output": []}
        self.validationDataset = {"input": [], "output": []}
        self.testingDataset = {"input": [], "output": []}

        # ==== initialize dataset ====
        self.__loadMeta()
        self.__createDataset()
        self.__preProcessing()

    def __preProcessing(self):
        # convert to np.array
        self.trainingDataset["input"] = np.array(self.trainingDataset["input"])
        self.validationDataset["input"] = np.array(self.validationDataset["input"])
        self.trainingDataset["output"] = np.array(self.trainingDataset["output"])
        self.validationDataset["output"] = np.array(self.validationDataset["output"])

        # extend dataset to have enough dim for conv2D
        self.trainingDataset["input"] = np.expand_dims(self.trainingDataset["input"], axis=-1)
        self.validationDataset["input"] = np.expand_dims(self.validationDataset["input"], axis=-1)

    def __loadMeta(self):
        """ Load the metadata for all subset of the DCASE2018 task4 dataset"""
        def load(meta_dir: str):
            if meta_dir != "":
                with open(meta_dir) as f:
                    data = f.readlines()

                return [d.split("\t") for d in data[1:]]

        self.metadata["weak"] = load(self.meta_train_weak)
        self.metadata["uid"] = load(self.meta_train_uid)
        self.metadata["uod"] = load(self.meta_train_uod)
        self.metadata["test"] = load(self.meta_test)

    def __createDataset(self):
        nbError = 0
        error_files = []

        # shuffle the metadata for the weak dataset
        shuffle(self.metadata["weak"])
        cutIndex = int(len(self.metadata["weak"]) * self.validationPercent)

        # ======== training subset ========
        training_data = self.metadata["weak"][cutIndex:]

        self.trainingDataset["input"] = []
        for info in training_data:
            path = os.path.join(self.feat_train_weak, info[0] + ".npy")
            if os.path.isfile(path):
                output = [0] * DCASE2018.NB_CLASS
                feature = np.load(path).T

                # save the original shape of the data
                if self.originalShape is None:
                    self.originalShape = feature.shape
                    print("original Shape: ", self.originalShape)

                self.trainingDataset["input"].append(feature)
                for cls in info[1].split(","): output[DCASE2018.class_correspondance[cls.rstrip()]] = 1
                self.trainingDataset["output"].append(output)
            else:
                nbError += 1
                error_files.append(path)

        # ======== validation subset ========
        validation_data = self.metadata["weak"][:cutIndex]

        self.validationDataset["input"] = []
        for info in validation_data:
            path = os.path.join(self.feat_train_weak, info[0] + ".npy")
            if os.path.isfile(path):
                output = [0] * DCASE2018.NB_CLASS
                feature = np.load(path).T

                self.validationDataset["input"].append(feature)
                for cls in info[1].split(","): output[DCASE2018.class_correspondance[cls.rstrip()]] = 1
                self.validationDataset["output"].append(output)
            else:
                nbError += 1
                error_files.append(path)

    def getInputShape(self):
        shape = self.trainingDataset["input"][0].shape
        return (shape[0], shape[1], 1)


if __name__=='__main__':
    dataset = DCASE2018(
        meta_train_weak="meta/weak.csv",
        feat_train_weak="/homeLocal/eriador/Documents/DCASE2018/task4/features/train/weak/mel",
    )

    # ==================================================================================================================
    #   Creating the model
    # ==================================================================================================================
    kInput = Input(dataset.getInputShape())

    block1 = Conv2D(filters=64, kernel_size=(3,3), padding="same")(kInput)
    block1 = BatchNormalization()(block1)
    block1 = Activation(activation="relu")(block1)
    block1 = MaxPooling2D(pool_size=(1, 4))(block1)
    block1 = Dropout(0.3)(block1)

    block2 = Conv2D(filters=64, kernel_size=(3,3), padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation(activation="relu")(block2)
    block2 = MaxPooling2D(pool_size=(1, 4))(block2)
    block2 = Dropout(0.3)(block2)

    block3 = Conv2D(filters=64, kernel_size=(3,3), padding="same")(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation(activation="relu")(block3)
    block3 = MaxPooling2D(pool_size=(1, 4))(block3)
    block3 = Dropout(0.3)(block3)
    # block3 ndim = 4

    targetShape = dataset.originalShape[0]
    reshape = Reshape(target_shape=(targetShape, 64))(block3)
    # reshape ndim = 3

    gru = Bidirectional(
        GRU(kernel_initializer='glorot_uniform', recurrent_dropout=0.0, dropout=0.3, units=64, return_sequences=True)
    )(reshape)
    print(gru.shape)

    output = TimeDistributed(
        Dense(dataset.nbClass, activation="sigmoid"),
    )(gru)
    print(output.shape)

    output = GlobalAveragePooling1D()(output)

    model = Model(inputs=kInput, outputs=output)
    keras.utils.print_summary(model, line_length=100)

    # compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # fit
    model.fit(
        x = dataset.trainingDataset["input"],
        y = dataset.trainingDataset["output"],
        epochs=100,
        batch_size=16,
        validation_data=(
            dataset.validationDataset["input"],
            dataset.validationDataset["output"]
        ),
        callbacks=[]
    )
