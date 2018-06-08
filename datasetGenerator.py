import os
from random import shuffle

import numpy as np

class DCASE2018:
    NB_CLASS = 11
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
        self.originalShape = None
        self.normalizer = normalizer

        if self.meta_train_uod != "":
            self.nbClass = nbClass + 1
        else:
            self.nbClass = nbClass

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
        # save original shape
        self.originalShape = self.trainingDataset["input"][0].shape

        # convert to np.array
        self.trainingDataset["input"] = np.array(self.trainingDataset["input"])
        self.validationDataset["input"] = np.array(self.validationDataset["input"])
        self.trainingDataset["output"] = np.array(self.trainingDataset["output"])
        self.validationDataset["output"] = np.array(self.validationDataset["output"])

        # normalization
        if self.normalizer is not None:
            print("==== Normalization stage ====")
            self.trainingDataset["input"] = self.normalizer.fit_transform(self.trainingDataset["input"])
            self.validationDataset["input"] = self.normalizer.fit_transform(self.validationDataset["input"])

        # extend dataset to have enough dim for conv2D
        self.trainingDataset["input"] = np.expand_dims(self.trainingDataset["input"], axis=-1)
        self.validationDataset["input"] = np.expand_dims(self.validationDataset["input"], axis=-1)


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

        def loadFeatures(subset: dict, toLoad: list, featDir: str):
            subset["input"] = []

            for info in toLoad:
                path = info[0]

                if os.path.isfile(path):
                    output = [0] * self.nbClass
                    feature = np.load(path)

                    subset["input"].append(feature)

                    if len(info) > 1:
                        for cls in info[1].split(","):
                            output[DCASE2018.class_correspondance[cls.rstrip()]] = 1
                    else:
                        output[DCASE2018.class_correspondance["blank"]] = 1

                    subset["output"].append(output)

        def balancedSplit():
            """ Split the weak subset into a balanced weak training and weak validation subsets"""
            splited = [[] for i in range(self.nbClass)]

            # separate the dataset into the 11 classes
            for info in self.metadata["weak"]:
                if len(info) > 1:
                    for cls in info[1].split(","):
                        splited[DCASE2018.class_correspondance[cls.rstrip()]].append(info)
                else:
                    splited[DCASE2018.class_correspondance["blank"]].append(info)

            # for each class, split into two (80%, 20%) for training and validation
            training = []
            validation = []
            for cls in splited:
                cutIndex = int(len(cls) * self.validationPercent)
                training.extend(cls[cutIndex:])
                validation.extend(cls[:cutIndex])

            return training, validation

        # convert basename to absolute path (needed for mixing both weak and uod dataset), and extend weak dataset
        # only if dir is used
        if self.meta_train_uod != "":
            for info in self.metadata["weak"]:
                info[0] = os.path.join(self.feat_train_weak, info[0] + ".npy")
            for info in self.metadata["uod"]:
                info[0] = os.path.join(self.feat_train_uod, info[0][:-1] + ".npy")

            self.metadata["weak"].extend(self.metadata["uod"])

        # split the weak subset into two classes wise evenly distributed
        training_data, validation_data = balancedSplit()

        # shuffle and load the features in memory
        shuffle(training_data)
        shuffle(validation_data)

        loadFeatures(self.trainingDataset, training_data, self.feat_train_weak)
        loadFeatures(self.validationDataset, validation_data, self.feat_train_weak)

    def getInputShape(self):
        shape = self.trainingDataset["input"][0].shape
        return (shape[0], shape[1], 1)

    def __str__(self):
        output = "-" * 30 + "\n"

        if self.meta_train_uod != "":
            output += "Dataset has been augmented using the unlabel out of domain for \"blank\" class \n"
            output += "%s files added\n\n" % len(self.metadata["uod"])

        output += "Training files: %s\nValidation files: %s\n" % (len(self.trainingDataset["input"]), len(self.validationDataset["input"]))
        output += "Validation ratio: %s" % self.validationPercent

        return output
