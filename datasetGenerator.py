import os
from random import shuffle

import numpy as np

class DCASE2018:
    NB_CLASS = 10
    class_correspondance = {
        "Alarm_bell_ringing": 0,
        "Speech": 1,
        "Dog": 2,
        "Cat": 3,
        "Vacuum_cleaner": 4,
        "Dishes": 5,
        "Frying": 6,
        "Electric_shaver_toothbrush": 7,
        "Blender": 8,
        "Running_water": 9,
        "blank": 10}

    def __init__(self,
                 feat_train_weak: str = "", feat_train_unlabelInDomain: str = "", feat_train_unlabelOutDomain: str = "",
                 meta_train_weak: str = "", meta_train_unlabelInDomain: str = "", meta_train_unlabelOutDomain: str = "",
                 feat_test: str = "", meta_test: str = "", validationPercent: float = 0.2, nbClass: int = 10,
                 nb_sequence: int = 1, normalizer = None):

        # directories
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
        self.normalizer = normalizer

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

        # normalization
        if self.normalizer is not None:
            print("==== Normalization stage ====")
            self.normalizer(self.trainingDataset["input"])
            self.normalizer(self.validationDataset["input"])

    def __loadMeta(self):
        """ Load the metadata for all subset of the DCASE2018 task4 dataset"""

        def load(meta_dir: str, nbFile: int = None):
            if meta_dir != "":
                with open(meta_dir) as f:
                    data = f.readlines()

                if nbFile is None:
                    return [d.split("\t") for d in data[1:]]
                else:
                    return [d.split("\t") for d in data[1:]][:nbFile]

        self.metadata["weak"] = load(self.meta_train_weak)
        self.metadata["uid"] = load(self.meta_train_uid)
        self.metadata["test"] = load(self.meta_test)

        # Use to extend training dataset, gather only 0.2 * len(training_dataset) of the uod
        nbFileForUod = len(self.metadata["weak"]) * 0.2
        self.metadata["uod"] = load(self.meta_train_uod, int(nbFileForUod))

    def __createDataset(self):
        nbError = 0
        error_files = []

        print("before extending the weak dataset")
        print(len(self.metadata["weak"]))

        # convert basename to absolute path (needed for mixing both weak and uod dataset, only if dir is used
        if self.meta_train_uod != "":
            self.metadata["weak"] = [os.path.join(self.feat_train_weak, info[0] + ".npy") for info in self.metadata["weak"]]
            self.metadata["weak"].extend([os.path.join(self.feat_train_uod, info[0] + ".npy") for info in self.metadata["uod"]])

        print("after extending the weak dataset")
        print(len(self.metadata["weak"]))

        # shuffle the metadata for the weak dataset
        shuffle(self.metadata["weak"])
        cutIndex = int(len(self.metadata["weak"]) * self.validationPercent)

        # ======== Training subset ========
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

                if len(info) > 1:
                    for cls in info[1].split(","):
                        output[DCASE2018.class_correspondance[cls.rstrip()]] = 1
                else:
                    output[DCASE2018.class_correspondance["blank"]] = 1

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

                if len(info) > 1:
                    for cls in info[1].split(","):
                        output[DCASE2018.class_correspondance[cls.rstrip()]] = 1

                else:
                    output[DCASE2018.class_correspondance["blank"]] = 1

                self.validationDataset["output"].append(output)

            else:
                nbError += 1
                error_files.append(path)

    def getInputShape(self):
        shape = self.trainingDataset["input"][0].shape
        return (shape[0], shape[1], 1)

