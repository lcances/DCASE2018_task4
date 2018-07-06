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
from sklearn.metrics import recall_score, precision_score, f1_score


class Binarizer(object):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Binarizer, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.thresholds = dict()
        self.optimized = False

        self.__initThresholds()


    def __initThresholds(self):
        """
        For each class, set the threshold to 0.5
        """
        for key in DCASE2018.class_correspondance:
            self.thresholds[key] = 0.5


    def optimize(self, y_true: np.array, predictionResult: np.array, method: str = "metrics"):
        """ Find the best thresholds for each classes with different methods available.

        methods available are
            - "metrics" -> use the precision, recall and f1 score to set the "optimized thresholds"
            - "auc" -> use the Aera Under the Curve methods to set the optimized thresholds

        :param y_true: ground truth
        :param predictionResult: 2-dimension numpy array not binarized
        :param method: methods to use for the threshold optimization: "metrics" | "auc"
        """
        if method == "metrics": optimizer = self.__metricsOptimization
        elif method == "auc": optimizer = self.__aucOptimization
        else:
            # TODO change sys.exit by raise
            print("Can't binarize on a array of dimension different that 2 or 3")
            sys.exit(1)

        if self.optimized == True:
            print("Binarizer was already optimized. Reseting the thresholds and perform new optimization")
            self.resetOptimization()

        optimizer(y_true, predictionResult)
        self.optimized = True

    def __metricsOptimization(self, y_true: np.array , predictionResult: np.array):
        binPrediction = self.binarize(predictionResult)

        f1 = f1_score(y_true, binPrediction, average=None)
        recall = recall_score(y_true, binPrediction, average=None)
        precision = precision_score(y_true, binPrediction, average=None)

        for cls in DCASE2018.class_correspondance:
            index = DCASE2018.class_correspondance[cls]

            self.thresholds[cls] = (f1[index] + recall[index] + precision[index]) / 3

    def __aucOptimization(self, y_true: np.array, predictionResult: np.array):
        pass

    def resetOptimization(self):
        """
        Cancel the optimization results and reset the thresholds to 0.5
        """
        self.__initThresholds()
        self.optimized = False

    def binarize(self, predictionResult: np.array) -> np.array:
        """
        Binarize the prediction results given using the defines thresholds, Can work with global prediction and
        temporal prediction
        :param predictionResult: 2 or 3 dimensions numpy array not binarized
        :return: 2 or 3 dimension numpy array representing the binarized prediction
        """
        if len(predictionResult.shape) == 2: return self.__globalBinarization(predictionResult)
        elif len(predictionResult.shape) == 3: return self.__temporalBinarization(predictionResult)
        else:
            # TODO change sys.exit by raise
            print("Can't binarize on a array of dimension different that 2 or 3")
            sys.exit(1)

    def __globalBinarization(self, predictionResult: np.array) -> np.array:
        output = []
        mappedScore = np.nan_to_num([self.thresholds[key] for key in self.thresholds])
        for i in range(len(predictionResult)):
            line = copy.copy(predictionResult[i])

            line[line > mappedScore] = 1
            line[line <= mappedScore] = 0
            output.append(line)

        return np.array(output)

    def __temporalBinarization(self, temporalPrediction: np.array) -> np.array:
        output = []
        mappedScore = np.nan_to_num([self.thresholds[key] for key in self.thresholds])

        for clip in temporalPrediction:
            bin = []

            for i in range(len(clip)):
                line = copy.copy(clip[i])

                line[line > mappedScore] = 1
                line[line <= mappedScore] = 0
                bin.append(line)

            output.append(bin)

        return np.array(output)

if __name__=="__main__":
    import random
    # create fake data (global prediction)
    def fakeGlobalPrediction():
        prediction = []
        for i in range(1000):
            score = [random.random() for i in range(10)]
            prediction.append(score)

        prediction = np.array(prediction)
        print(prediction.shape)
        print(prediction)

        b = Binarizer()
        binPrediction = b.binarize(prediction)
        print(binPrediction)
        print(binPrediction.shape)

    # create fake data (temporal prediction)
    def fakeTemporalPrediction():
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
        binPrediction = b.binarize(prediction)
        print(binPrediction)
        print(binPrediction.shape)

    fakeTemporalPrediction()


