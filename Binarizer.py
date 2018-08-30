"""
Binarizer is implemented under the form of a Singleton and therefore can be use between the different
module and stage of the system without having to keep trace of it. The __new__ methods have been
overwritten in order to always return the only one object that should be present.

Allow the binarization of the prediction results using global thresholds or optimized one.
These thresholds are set to 0.5 as default and can be optimized using AUC score
"""
from datasetGenerator import DCASE2018
import numpy as np
import copy
import sys
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve


class Binarizer(object):
    _instance = None
    _exist = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Binarizer, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not Binarizer._exist:
            self.thresholds = dict()
            self.optimized = False

            self.__init_thresholds()
            Binarizer._exist = True

    def __init_thresholds(self):
        """
        For each class, set the threshold to 0.5
        """
        for key in DCASE2018.class_correspondance:
            self.thresholds[key] = 0.5

    def optimize(self, y_true: np.array, prediction_result: np.array, method: str = "simulated_annealing"):
        """ Find the best thresholds for each classes with different methods available.

        methods available are
            - "metrics" -> use the precision, recall and f1 score to set the "optimized thresholds"
            - "auc" -> use the Aera Under the Curve methods to set the optimized thresholds

        :param y_true: ground truth
        :param prediction_result: 2-dimension numpy array not binarized
        :param method: methods to use for the threshold optimization: "metrics" | "auc"
        """
        _method = ["metrics", "auc", "simulated_annealing"]
        if method == _method[0]:
            optimizer = self.__metrics_optimization
        elif method == _method[1]:
            optimizer = self.__aucOptimization
        elif method == _method[2]:
            optimizer = self.__simulated_annealing_optimization
        else:
            # TODO change sys.exit by raise
            print("Can't binarize on a array of dimension different that 2 or 3")
            sys.exit(1)

        if self.optimized:
            print("Binarizer was already optimized. Reseting the thresholds and perform new optimization")
            self.resetOptimization()

        optimizer(y_true, prediction_result)
        self.optimized = True

    def __simulated_annealing_optimization(self, y_true, prediction_result: np.array):
        def apply_threshold(thresh: list, prediction: np.array) -> list:
            pred = copy.copy(prediction)

            for clsInd in range(len(pred.T)):
                pred.T[clsInd][pred.T[clsInd] > thresh[clsInd]] = 1
                pred.T[clsInd][pred.T[clsInd] <= thresh[clsInd]] = 0

            return pred

        def init_thresholds():
            return np.array([random.randint(40, 60) / 100 for _ in range(10)])

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        def calc_delta(thresholds, weight):
            output = []
            for th in thresholds:
                g = gaussian(th, 0.5, 0.08)
                output.append((random.random() * 2 * g - g) * weight)
            return np.array(output)

        # initialization
        thresholds = [0.5 for _ in range(10)]
        pred0 = apply_threshold(thresholds, prediction_result)
        f10 = f1_score(pred0, y_true, average=None)

        best = {"thresholds": thresholds, "mean f1": f10.mean(), "f1": f10}

        meta_iter = 100
        nb_iter = 30

        for j in range(meta_iter):
            thresholds = init_thresholds()
            weight = 0.07
            decay = weight / (nb_iter * 1.9)

            for i in range(nb_iter):
                delta = calc_delta(thresholds, weight)
                thresholds += delta
                weight -= decay

                nPred = apply_threshold(thresholds, prediction_result)
                f1 = f1_score(nPred, y_true, average=None)

                if f1.mean() > best["mean f1"]:
                    best["thresholds"] = thresholds
                    best["mean f1"] = f1.mean()
                    best["f1"] = f1

        cpt = 0
        for key in self.thresholds:
            self.thresholds[key] = best["thresholds"][cpt]
            cpt += 1

    def __metrics_optimization(self, y_true: np.array, prediction_result: np.array):
        bin_prediction = self.binarize(prediction_result)

        f1 = f1_score(y_true, bin_prediction, average=None)
        recall = recall_score(y_true, bin_prediction, average=None)
        precision = precision_score(y_true, bin_prediction, average=None)

        for cls in DCASE2018.class_correspondance:
            index = DCASE2018.class_correspondance[cls]

            self.thresholds[cls] = (f1[index] + recall[index] + precision[index]) / 3

    def __aucOptimization(self, y_true: np.array, prediction_result: np.array):
        fpr = dict()
        tpr = dict()

        for cls in DCASE2018.class_correspondance:
            y_true = np.array(y_true[:, cls])
            y_pred = np.array(prediction_result[:, cls])

            fpr[cls], tpr[cls], thresholds = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)

            best_th_ind = np.argmax(tpr[cls] - fpr[cls])
            self.thresholds[cls] = thresholds[best_th_ind]

    def resetOptimization(self):
        """
        Cancel the optimization results and reset the thresholds to 0.5
        """
        self.__init_thresholds()
        self.optimized = False

    def binarize(self, prediction_result: np.array) -> np.array:
        """
        Binarize the prediction results given using the defines thresholds, Can work with global prediction and
        temporal prediction
        :param prediction_result: 2 or 3 dimensions numpy array not binarized
        :return: 2 or 3 dimension numpy array representing the binarized prediction
        """
        if len(prediction_result.shape) == 2: return self.__global_binarization(prediction_result)
        elif len(prediction_result.shape) == 3: return self.__temporal_binarization(prediction_result)
        else:
            # TODO change sys.exit by raise
            print("Can't binarize on a array of dimension different that 2 or 3")
            sys.exit(1)

    def __global_binarization(self, prediction_result: np.array) -> np.array:
        output = []
        mappedScore = np.nan_to_num([self.thresholds[key] for key in self.thresholds])
        for i in range(len(prediction_result)):
            line = copy.copy(prediction_result[i])

            line[line > mappedScore] = 1
            line[line <= mappedScore] = 0
            output.append(line)

        return np.array(output)

    def __temporal_binarization(self, temporal_prediction: np.array) -> np.array:
        output = []
        mapped_score = np.nan_to_num([self.thresholds[key] for key in self.thresholds])

        for clip in temporal_prediction:
            bin = []

            for i in range(len(clip)):
                line = copy.copy(clip[i])

                line[line > mapped_score] = 1
                line[line <= mapped_score] = 0
                bin.append(line)

            output.append(bin)

        return np.array(output)


if __name__ == "__main__":
    import random

    # create fake data (global prediction)
    def fake_global_prediction():
        prediction = []
        for i in range(1000):
            score = [random.random() for i in range(10)]
            prediction.append(score)

        prediction = np.array(prediction)
        print(prediction.shape)
        print(prediction)

        b = Binarizer()
        bin_prediction = b.binarize(prediction)
        print(bin_prediction)
        print(bin_prediction.shape)

    # create fake data (temporal prediction)
    def fake_temporal_prediction():
        prediction = []
        for i in range(1000):
            clip = []
            for j in range(500):
                score = [random.random() for k in range(10)]
                clip.append(score)
            prediction.append(clip)

        prediction = np.array(prediction)
        print(prediction)
        print(prediction.shape)

        b = Binarizer()
        bin_prediction = b.binarize(prediction)
        print(bin_prediction)
        print(bin_prediction.shape)

    fake_temporal_prediction()

