from keras.callbacks import ProgbarLogger, Callback, ModelCheckpoint

"""
{'batch_size': 8, 'epochs': 200, 'steps': None, 'samples': 1237, 'verbose': 0, 'do_validation': True, 'metrics': ['loss', 'binary_accuracy', 'precision', 'recall', 'f1', 'val_loss', 'val_binary_accuracy', 'val_precision', 'val_recall', 'val_f1']}
starting

"""

class MyProgbarLogger(Callback):
    def __init__(self):
        super().__init__()

        self.firstLineWritten = False

    def on_batch_begin(self, batch, logs=None):
        super().on_batch_begin(batch, logs)

    # ======== BATCH ========
    def on_batch_end(self, batch, logs=None):
        #super().on_batch_end(batch, logs)
        if not self.firstLineWritten:
            # print two first column
            print("{:<7}".format("batch"), end="")
            print("{:<7}".format("progr"), end="")

            # print metrics name
            for key, value in logs.items():
                if (key in self.params["metrics"]):
                    print("{:<10}".format(key[:8]), end="")
            print("\n")
            print("-"*10*len(logs))

            self.firstLineWritten = True

        else:
            print("{:<7}".format(logs["batch"]), end="")
            print("%{:<7}".format(str(int(logs["batch"]) * int(self.params["batch_size"]) / self.params["samples"] * 100)[:3]), end="")

            for key, value in logs.items():
                if key in self.params["metrics"]:
                    print("{:<10}".format(str(value)[:6]), end="")
            print("", end="\r")


    # ======== EPOCH ========
    def on_epoch_end(self, epoch, logs=None):
        #super().on_epoch_end(epoch, logs)
        print(logs)
