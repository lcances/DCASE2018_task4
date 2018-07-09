import sys
import numpy as np
import copy

from Binarizer import Binarizer
from datasetGenerator import DCASE2018

class Encoder:
    def __init__(self):
        self.frameLength = 0
        self.nbFrame = 0

    def encode(self, temporalPrediction: np.array, method: str = "threshold", **kwargs) -> str:
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
        _methods=["threshold", "hysteresis", "derivative"]
        if method not in _methods:
            print("method %s doesn't exist. Only", _methods, " available")
            sys.exit(1)

        if method == _methods[0]: encoder = self.__encodeUsingThreshold
        elif method == _methods[2]: encoder = self.__encodeUsingDerivative
        elif method == _methods[1]: encoder = self.__encodeUsingHysteresis
        else:
            sys.exit(1)

        self.nbFrame = temporalPrediction.shape[1]
        self.frameLength = 10 / self.nbFrame
        return encoder(temporalPrediction, **kwargs)

    def __encodeUsingHysteresis(self, temporalPrediction: np.array, **kwargs) -> list:
        """ Hysteresys based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :param kwargs: Extra arguments - "high" and "low" (thresholds for the hysteresis)
        :return: the result of the system under the form of a strong annotation text where each line represent on timed event
         """
        low = kwargs["low"] if "low" in kwargs.keys() else 0.4
        high = kwargs["high"] if "high" in kwargs.keys() else 0.6
        prediction = temporalPrediction

        output = []

        for clip in prediction:
            labeled = dict()

            cls = 0
            for predictionPerClass in clip.T:
                converted = list()
                segment = [0, 0]
                nbSegment = 1
                for i in range(len(predictionPerClass)):
                    element = predictionPerClass[i]

                    # first element
                    if i == 0:
                        segment = [1.0, 1] if element > high else [0.0, 1]

                    # then
                    if element > high and segment[0] == 1:
                        segment[1] += 1

                    elif element > high and segment[0] == 0:
                        converted.append(segment)
                        nbSegment += 1
                        segment = [1.0, 1]

                    elif element <= low and segment[0] == 0:
                        segment[1] += 1

                    elif element <= low and segment[0] == 1:
                        converted.append(segment)
                        nbSegment += 1
                        segment = [0.0, 0]

                if nbSegment == 1:
                    converted.append(segment)

                labeled[cls] = copy.copy(converted)
                cls += 1

            output.append(labeled)

        return output

    def __encodeUsingThreshold(self, temporalPrediction: np.array, **kwargs) -> list:
        """ Threshold based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :param kwargs: Extra arguments. None possible in this method
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """
        output = []
        temporalPrecision = 200        # ms

        # binarize the results using the thresholds (default or optimized) provided by the Binarizer
        binarizer = Binarizer()
        binPrediction = binarizer.binarize(temporalPrediction)

        # Merging "hole" that are smeller than 200 ms
        stepLength = DCASE2018.CLIP_LENGTH / temporalPrediction.shape[1] * 1000     # in ms
        maxHoleSize = int(temporalPrecision / stepLength)

        for clip in binPrediction:
            labeled = dict()

            cls = 0
            for binPredictionPerClass in clip.T:
                # convert the binarized list into a list of tuple representing the element and it's number of
                # occurrence. The order is conserved and the total sum should be equal to 10s

                # first pass --> Fill the holes
                for i in range(len(binPredictionPerClass) - maxHoleSize):
                    window = binPredictionPerClass[i : i+maxHoleSize]

                    if window[0] == window[-1] == 1:
                        window[:] = [window[0]] * maxHoleSize

                # second pass --> split into segments
                converted = []
                cpt = 0
                nbSegment = 0
                previousElt = None
                for element in binPredictionPerClass:
                    if previousElt is None:
                        previousElt = element
                        cpt += 1
                        nbSegment = 1
                        continue

                    if element == previousElt:
                        cpt += 1

                    else:
                        converted.append((previousElt, cpt))
                        previousElt = element
                        nbSegment += 1
                        cpt = 1

                # case where the class is detect during the whole clip
                if nbSegment == 1:
                    converted.append((previousElt, cpt))

                labeled[cls] = copy.copy(converted)
                cls += 1

            output.append(labeled)

        return output

    def __encodeUsingDerivative(self, temporalPrediction: np.array, **kwargs) -> list:
        """ Threshold based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """

        def futureIsFlat(prediction: np.array, currentPos: int, flat: float = 0.05, window_size: int = 5) -> bool:
            """
            Detect what is following is "kinda" flat.
            :param prediction: The prediction values of the current class
            :param currentPos: The current position of the window (left side)
            :return: True is the near future of the curve is flat, False otherwise
            """
            slopes = 0

            # if not future possible (end of the curve)
            if (currentPos + window_size) > len(prediction):
                return False

            # sum the slope value for the next <window_size> window
            for i in range(currentPos, currentPos + window_size):
                window = prediction[i:i+window_size]
                slopes += window[-1] - window[0]

            averageSlope = slopes / window_size

            # is approximately flat, the return True, else False
            return abs(averageSlope) < flat

        # retreive the argument from kwargs
        keys = kwargs.keys()
        rising = kwargs["rising"] if "rising" in keys else 0.5
        decreasing = kwargs["decreasing"] if "decreasing" in keys else -0.5
        flat = kwargs["flat"] if "flat" in keys else 0.05
        window_size = kwargs["window_size"] if "window_size" in keys else 5
        high = kwargs["high"] if "high" in keys else 0.5

        output = []

        for clip in temporalPrediction:
            cls = 0
            labeled = dict()

            for predictionPerClass in clip.T:

                nbSegment = 1
                segments = []
                segment = [0.0, 0]
                for i in range(len(predictionPerClass) - window_size):
                    window = predictionPerClass[i:i+window_size]

                    slope = window[-1] - window[0]

                    # first element
                    if i == 0:
                        segment = [1.0, 1] if window[0] > high else [0.0, 1]

                    # rising slope while on "low" segment --> changing segment
                    if slope > rising and segment[0] == 0:
                        segments.append(segment)
                        nbSegment += 1
                        segment = [1.0, 1]

                    # rising slope while on "high" segment --> same segment
                    elif slope > rising and segment[0] == 1:
                        segment[1] += 1

                    # decreasing slope while on "low" segment --> same segment
                    elif slope < decreasing and segment[0] == 0:
                        segment[1] += 1

                    # decreasing slope while on "high" segment --> one extra condition, future is flat ?
                    elif slope < decreasing and segment[0] == 1:
                        # is there is no flat plateau right after --> same segment
                        if not futureIsFlat(predictionPerClass, i, flat, window_size):
                            segment[1] += 1

                        # Otherwise --> change segment
                        else:
                            segments.append(segment)
                            nbSegment += 1
                            segment = [0.0, 1]

                if nbSegment == 1:
                    segments.append(copy.copy(segment))

                labeled[cls] = segments
                cls += 1

            output.append(labeled)
        return output



    def parse(self, allSegments: list, testFilesName: list) -> str:
        output = ""

        for clipIndex in range(len(allSegments)):
            clip = allSegments[clipIndex]

            for cls in clip:
                start = 0
                for segment in clip[cls]:
                    if segment[0] == 1.0:
                        output += "%s\t%f\t%f\t%s\n" % (
                            testFilesName[clipIndex],
                            start * self.frameLength,
                            (start + segment[1]) * self.frameLength,
                            DCASE2018.class_correspondance_reverse[cls]
                        )
                    start += segment[1]

        return output

if __name__=='__main__':
    import random
    e = Encoder()

    # create fake data (temporal prediction)
    def mRandom():
        r = random.random()
        return r

    def fakeTemporalPrediction():
        prediction = []
        for i in range(10):
            clip = []
            for j in range(200):
                score = [mRandom() for k in range(10)]
                clip.append(score)
            prediction.append(clip)

        prediction = np.array(prediction)

        #o = e.encode(prediction)       # basic thresold with hold filling
        o = e.encode(prediction, method="derivative")
        for k in o:
            print(len(k[0]), k[0])
        t = e.parse(o, prediction[:,0,0])


    fakeTemporalPrediction()