from keras.callbacks import Callback
from sklearn.metrics import classification_report

"""
{'batch_size': 8, 'epochs': 200, 'steps': None, 'samples': 1237, 'verbose': 0, 'do_validation': True, 'metrics': ['loss', 'binary_accuracy', 'precision', 'recall', 'f1', 'val_loss', 'val_binary_accuracy', 'val_precision', 'val_recall', 'val_f1']}
starting

"""

class CompleteLogger(Callback):
    def __init__(self, logPath:str, val_true):
        super().__init__()

        self.val_true = val_true

        self.trainMetrics = []
        self.validationMetrics = []

        self.logPath = logPath
        self.logFile = None
        self.currentEpoch = 0

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        # open the log file
        if self.logPath is not None:
            self.logFile = open(self.logPath, "w")

        # Split the complete list of metrics into two separated list. training and validation (left and right)
        middle = int(len(self.params["metrics"]) / 2)
        self.trainMetrics = self.params["metrics"][:middle]
        self.validationMetrics = self.params["metrics"][middle:]

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
        print("_" * 6 * (len(self.trainMetrics) + len(self.validationMetrics) + 18))

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if self.logPath is not None:
            self.logFile.close()

    def on_batch_end(self, batch, logs=None):
        """
        at every batch, the display is updated for the training part of the table
        """

        print("{:<8}".format(self.currentEpoch), end="")
        print("%{:<10}".format(str(int(logs["batch"]) * int(self.params["batch_size"]) / self.params["samples"] * 100)[:3]), end="")

        for m in self.trainMetrics:
            print("{:<12}".format(str(logs[m])[:6], end=""))

        print("", end="\r")

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        self.currentEpoch += 1

    def on_epoch_end(self, epoch, logs=None):
        """
        Add the saving of the f1 measure for each class separately on a file called: metrics_learn
        """
        super().on_epoch_end(epoch, logs)

        # display all metrics (tra and val)
        print("{:<7}".format(logs["batch"]), end="")
        print("%{:<7}".format(str(int(logs["batch"]) * int(self.params["batch_size"]) / self.params["samples"] * 100)[:3]), end="")

        for m in self.trainMetrics:
            print("{:<12}".format(str(logs[m])[:6], end=""))

            print(" | ", end="")
            for m in self.validationMetrics:
                print("{:<12}".format(str(logs[m])[:6]), end="")


        # for the detail of the metrics, it will be save into the log file
        if self.logPath is not None:
            val_pred = self.model.predict(self.val_true)

            self.logFile.write("%s\n" % epoch)
            self.logFile.write(classification_report(self.val_true, val_pred))


