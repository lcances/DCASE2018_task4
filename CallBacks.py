from keras.callbacks import Callback
from keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score

from datasetGenerator import DCASE2018
from Binarizer import Binarizer

import signal, os, sys, time

class CompleteLogger(Callback):
    def __init__(self, logPath: str, validation_data: tuple, history_size: int = 10,
            fallback: bool = False, fallBackThreshold: int = 5, stopAt: int = 100,
            display: bool = True
            ):

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
        self.stopAt = stopAt
        self.epochStart = 0
        self.epochDuration = 0

        self.history_size = history_size
        self.history = []       # a history of the best models
        self.sortedHistory = []
        self.f1History = {"train": [], "val": []}

        self.fallbackCooldown = self.history_size
        self.fallbackTh = fallBackThreshold
        self.nbFallback = 0
        self.cooldown = 0
        self.coolingDown = False
        self.nbMaxFallback = 3

        self.transferMode = False
        self.display = display
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


#        if self.currentEpoch == 35:
#            currentLr = K.get_value(self.model.optimizer.lr)
#            K.set_value(self.model.optimizer.lr, currentLr / 2)

        self.epochStart = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """
        Add the saving of the f1 measure for each class separately on a file called: metrics_learn
        """
        super().on_epoch_end(epoch, logs)
        self.epochDuration = time.time() - self.epochStart

        self.__computeMetrics()
        self.__toHistory(logs)

        self.__printMetrics(logs, validation=True, overwrite=False)
        self.__logGeneralEpoch(logs)
        self.__logClassesEpoch()

        self.__fallingBack()

        # stop training if stopAt is reached
        if self.currentEpoch == self.stopAt:
            self.model.stop_training = True

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
    #       HISTORY AND FALLBACK FUNCTION
    # ==================================================================================================================
    def __toHistory(self, logs=None):
        average_f1 = float(logs[self.validationMetrics[-1]])

        # add the current model to the list of history
        self.history.append(
                {"weights": self.model.get_weights(),
                 "average f1": average_f1,
                 "epoch": self.currentEpoch}
                )
        self.sortedHistory.append(
                {"weights": self.model.get_weights(),
                 "average f1": average_f1,
                 "epoch": self.currentEpoch}
                )
        self.f1History["train"].append(float(logs[self.trainMetrics[-1]]))
        self.f1History["val"].append(float(logs[self.validationMetrics[-1]]))


        # sort the list using the average f1 key
        self.sortedHistory = sorted(self.history, key=lambda k: k['average f1'])

        # keep only the <history_size> first
        self.history = self.history[:self.history_size]
        self.sortedHistory = self.sortedHistory[-self.history_size:]
        self.f1History["train"] = self.f1History["train"][:self.history_size]
        self.f1History["val"] = self.f1History["val"][:self.history_size]

    def __cutHistoryTo(self, toCut: int):
        self.history = self.history[:toCut]
        self.f1History["train"] = self.f1History["train"][:toCut]
        self.f1History["val"] = self.f1History["val"][:toCut]

    def __fallingBack(self):
        # managing cooling down
        if self.coolingDown:
            self.cooldown += 1

            if self.cooldown > self.fallbackCooldown:
                self.cooldown = 0
                self.coolingDown = False

            print("cooling down")
            return;

        # if too much fallingback
        if self.nbFallback == self.nbMaxFallback:
            return;

        # if not enough time spent
        if self.currentEpoch < self.history_size:
            return;

        curF1Val = self.f1History["val"][-1]
        curF1Tra = self.f1History["train"][-1]
        diff = curF1Tra - curF1Val

        if diff > self.fallbackTh:

            betterEpoch = self.history_size - 1
            mini = diff
            for i in range(self.history_size - 1, -1, -1):
                cDiff = self.f1History["train"][i] - self.f1History["val"][i]
                if cDiff < mini:
                    mini = cDiff
                    betterEpoch = i

            print("Overfitting... going back in time %s epochs behind" % (betterEpoch))
            print("train, val diff: %.2f" % mini)
            self.model.set_weights(self.history[betterEpoch]["weights"])
            self.nbFallback += 1

            # pruning history
            cutoff = self.currentEpoch - (self.history_size - betterEpoch)
            self.__cutHistoryTo(cutoff)

            # starting cooldown
            self.coolingDown = True


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
        if self.display:
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
        if self.display:
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

    # ==================================================================================================================
    #       EARLY KILL HANDLER
    # ==================================================================================================================
    def __exitAndSave(self, signum, frame):
        print("SIGTERM OR SIGKILL SIGNAL RECEIVED...")
        print("Saving the best model to early_stop_weights.h5")

        model.set_weights(sortedHistory[-1]["weights"])
        model.save_weights("early_stop_weights.h5py")

        sys.exit(2)


