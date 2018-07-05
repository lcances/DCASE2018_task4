import sys
import numpy as np

from Binarizer import Binarizer

class Encoder:
    def __init__(self):
        pass

    def encode(self, temporalPrediction: np.array, method: str = "threshold") -> str:
        """
        Perform the localization of the sound event present in the file.

        Using the temporal prediction provided y the last step of the system, it will "localize" the sound event
        inside the file under the form of a strongly annotated line. (see DCASE2018 task 4 strong label exemple).
        There is two methods implemented here, one using a simple threshold based segmentation and an other using
        a modulation system based on the variance of the prediction over the time.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """
        # parameters verification
        if method != "threshold" and method != "modulation":
            print("method %s doesn't exist. Only \"threshold\" and \"modulation\" exist")
            sys.exit(1)

        if method == "threshold":
            return self.__encodeUsingThreshold(temporalPrediction)
        else:
            return self.__encodeUsingModulation(temporalPrediction)

    def __encodeUsingThreshold(self, temporalPrediction: np.array) -> str:
        """ Threshold based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """
        output = str()

        # binarize the results using the thresholds (default or optimized) provided by the Binarizer
        binarizer = Binarizer()
        binPrediction = binarizer.binarize(temporalPrediction)

        #



    def __encodeUsingModulation(self, temporalPrediction: np.array):
        """ modulation based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """
        pass
