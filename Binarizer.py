"""
TODO
"""
__author__ = "Léo Cances"
__copyright__ = ""
__credits__ = ["Léo Cances"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Léo Cances"
__email__ = "leocances@gmail.com"
__status__ = "Production"

from datasetGenerator import DCASE2018
import numpy as np
import copy

class Binarizer:
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


    def optimize(self, predictionResult: np.array):
        """
        Find the best thresholds for each classes using area under the curve and ROC curves
        :param predictionResult: 2-dimension numpy array not binarized
        """
        self.optimized = True

    def resetOptimization(self):
        """
        Cancel the optimization results and reset the thresholds to 0.5
        """
        self.__initThresholds()
        self.optimized = False

    def binarize(self, predictionResult: np.array) -> np.array:
        """
        Binarize the prediction results given using the defines thresholds, Each column represent one class.
        :param predictionResult: 2-dimensions numpy array not binarized
        :return: 2-dimension numpy array representing the binarized prediction
        """
        output = []
        mappedScore = np.nan_to_num([self.thresholds[key] for key in self.thresholds])
        for i in range(len(predictionResult)):
            line = copy.copy(predictionResult[i])

            line[line > mappedScore] = 1
            line[line < mappedScore] = 0
            output.append(line)

        return np.array(output)