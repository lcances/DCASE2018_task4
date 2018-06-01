from keras.callbacks import Callback, BaseLogger, ProgbarLogger

class MyCallBack(Callback):
    def __init__(self):
        Callback.__init__(self)

    def on_epoch_end(self, epoch, logs=None):
        """
        Perform a validation step and compute the differents metrics
        :param epoch:
        :param logs:
        :return:
        """
        super().on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)

