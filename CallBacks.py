from keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score

import time
import os
from datasetGenerator import DCASE2018
from Binarizer import Binarizer


class CompleteLogger(Callback):
    def __init__(self, logPath: str, validation_data: tuple, history_size: int = 3):
        super().__init__()

        self.validation_input = validation_data[0]
        self.validation_output = validation_data[1]

        self.metrics = []
        self.trainMetrics = []
        self.validationMetrics = []

        self.logging = logPath is not None
        self.logPath = {"general": logPath}
        self.logFile = {}

        self.currentEpoch = 0
        self.epochStart = 0
        self.epochDuration = 0

        self.history_size = history_size
        self.history = []       # a history of the best models

        self.transferMode = False
        self.binarizer = Binarizer()

    def toggleTransfer(self):
        self.transferMode != self.transferMode

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        self.__initLogFiles()

        # Split the complete list of metrics into two separated list. training and validation (left and right)
        middle = int(len(self.params["metrics"]) / 2)
        self.metrics = self.params["metrics"]
        self.trainMetrics = self.params["metrics"][:middle]
        self.validationMetrics = self.params["metrics"][middle:]

        self.__printHeader()
        self.__logGeneralHeader()

    def on_train_end(self, logs=None):
        super().on_train_end(logs)

        if self.logPath is not None:
            self.__finishLogFiles()

    def on_batch_end(self, batch, logs=None):
        self.__printMetrics(logs)

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        self.currentEpoch += 1

        self.epochStart = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """
        Add the saving of the f1 measure for each class separately on a file called: metrics_learn
        """
        super().on_epoch_end(epoch, logs)
        self.epochDuration = time.time() - self.epochStart

        self.__computeMetrics()
        self.__toHistory()

        self.__printMetrics(logs, validation=True, overwrite=False)
        self.__logGeneralEpoch(logs)
        self.__logClassesEpoch()

    # ==================================================================================================================
    #       Classes metrics compute
    # ==================================================================================================================
    def __computeMetrics(self):
        # compute the results of the metrics and log them to the files

        # calc metrics for each classes separately
        prediction = self.model.predict(self.validation_input)
        prediction = self.binarizer.binarize(prediction)

        self.precision = precision_score(self.validation_output, prediction, average=None)
        self.recall = recall_score(self.validation_output, prediction, average=None)
        self.f1 = f1_score(self.validation_output, prediction, average=None)


    # ==================================================================================================================
    #       LOG FUNCTIONS
    # ==================================================================================================================
    def __toHistory(self):
        average_f1 = self.f1.mean()

        # add the current model to the list of history
        self.history.append( {"weights": self.model.get_weights(), "average f1": average_f1, "epoch": self.currentEpoch} )

        # sort the list using the average f1 key
        self.history = sorted(self.history_size, key=lambda k: k['average f1'])

        # keep only the <history_size> first
        self.history = self.history[:self.history_size]




    # ==================================================================================================================
    #       LOG FUNCTIONS
    # ==================================================================================================================
    def __initLogFiles(self):
        if not self.logging:
            return

        dirPath = self.logPath["general"]
        dirName = os.path.dirname(dirPath)
        fileName = os.path.basename(dirPath)

        if not os.path.isdir(dirName):
            print("doesn't exist, creating it")
            os.makedirs(dirName)

        # add one files for each metrics that will be compute for classes
        self.logPath["precision"] = dirPath + "_precision.csv"
        self.logPath["recall"] = dirPath + "_recall.csv"
        self.logPath["f1"] = dirPath + "_f1.csv"
        self.logPath["general"] += "_metrics.csv"

        # open the files
        for key in self.logPath.keys():
            if not self.transferMode:
                self.logFile[key] = open(self.logPath[key], "w")
            else:
                self.logFile[key] = open(self.logPath[key], "a")

        # write headers
        if not self.transferMode:
            for key in self.logFile:
                if key != "general":
                    self.__logClassesheader(self.logFile[key])
            self.__logGeneralHeader()

    def __finishLogFiles(self):
        if not self.logging:
            return

        for key in self.logPath.keys():
            self.logFile[key].close()

    def __logGeneralHeader(self):
        if self.logging:
            self.logFile["general"].write("epoch,progress,")

            for m in self.metrics:
                self.logFile["general"].write("%s," % m)

            self.logFile["general"].write("duration\n")

    def __logGeneralEpoch(self, logs: dict):
        if self.logging:
            self.logFile["general"].write("%s,100," % (self.currentEpoch))

            # all metrics
            for m in self.metrics:
                if m != self.validationMetrics[-1]:
                    self.logFile["general"].write("%s," % str(logs[m])[:6])
                else:
                    self.logFile["general"].write("%s" % str(logs[m])[:6])

            self.logFile["general"].write("%s\n" % self.epochDuration)

    def __logClassesheader(self, file):
        if self.logging:
            file.write("epoch,")
            for key in DCASE2018.class_correspondance.keys():
                if key != list(DCASE2018.class_correspondance.keys())[-1]:
                    file.write("%s," % key)
                else:
                    file.write("%s\n" % key)

    def __logClassesEpoch(self):
        def convertToCSV(line: list):
            return ",".join(map(str, line))

        if self.logging:
            self.logFile["precision"].write(str(self.currentEpoch) + "," + convertToCSV(self.precision) + "\n")
            self.logFile["recall"].write(str(self.currentEpoch) + "," + convertToCSV(self.recall) + "\n")
            self.logFile["f1"].write(str(self.currentEpoch) + "," + convertToCSV(self.f1) + "\n")

    # ==================================================================================================================
    #       DISPLAY FUNCTIONS
    # ==================================================================================================================
    def __printMetrics(self, logs: dict, validation: bool = False, overwrite: bool = True):
        # two first column
        print("{:<8}".format(self.currentEpoch), end="")

        if "batch" in logs.keys():
            percent = int(logs["batch"]) * int(self.params["batch_size"]) / int(self.params["samples"]) * 100
            print("%{:<10}".format(str(int(percent))[:3]), end="")
        else:
            print("%{:<10}".format("100"), end="")

        for m in self.trainMetrics:
            print("{:<12}".format(str(logs[m])[:6]), end="")

        if validation:
            print(" | ", end="")
            for m in self.validationMetrics:
                print("{:<12}".format(str(logs[m])[:6]), end="")

        if overwrite:
            print("", end="\r")
        else:
            print(" %.2f" % self.epochDuration, end="")
            print("", end="\n")

    def __printHeader(self):
        # Print the complete header
        print("{:<8}".format("epoch"), end="")
        print("{:<10}".format("progress"), end="")

        # print training metric name
        for m in self.trainMetrics:
            print("{:<12}".format(m[:10]), end="")

        # print validation metric name
        print(" | ", end="")
        for m in self.validationMetrics:
            print("{:<12}".format(m[:10]), end="")

        print("")
        print("-" * (18 + 12 * len(self.trainMetrics) + 3 + 12 * len(self.validationMetrics)) )


