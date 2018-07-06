import sys
import numpy as np
import copy

from Binarizer import Binarizer
from datasetGenerator import DCASE2018

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

    def __encodeUsingThreshold(self, temporalPrediction: np.array) -> list:
        """ Threshold based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """
        output = []

        # binarize the results using the thresholds (default or optimized) provided by the Binarizer
        binarizer = Binarizer()
        binPrediction = binarizer.binarize(temporalPrediction)

        # Merging "hole" that are smeller than 200 ms
        stepLength = DCASE2018.CLIP_LENGTH / temporalPrediction.shape[1] * 1000     # in ms
        maxHoleSize = 200 / stepLength

        print("step length: ", stepLength)
        print("maxHoleSize: ", maxHoleSize)


        for clip in binPrediction:
            labeled = dict()

            cls = 0
            for binPredictionPerClass in clip.T:
                # convert the binarized list into a list of tuple representing the element and it's number of
                # occurrence. The order is conserved and the total sum should be equal to 10s
                converted = []
                cpt = 0
                previousElt = None
                for element in binPredictionPerClass:
                    if previousElt is None:
                        previousElt = element
                        cpt += 1
                        continue

                    if element == previousElt:
                        cpt += 1

                    else:
                        converted.append((previousElt, cpt))
                        previousElt = element
                        cpt = 1

                # "fill the hole"
                # TODO add the two exeception (start and end)
                compressed = []
                for segmentInd in range(len(converted) - 1):
                    segment = converted[segmentInd]
                    segmentSize = segment[1]

                    if segmentSize < maxHoleSize:
                        while segmentSize < maxHoleSize:
                            segmentSize += converted[segmentInd+1][1]
                            segmentInd *= 1

                        compressed.append( (segment[0], segmentSize) )
                    else:
                        compressed.append(segment)

                labeled[cls] = copy.copy(compressed)

            output.append(labeled)

        return output

    def __encodeUsingModulation(self, temporalPrediction: np.array):
        """ modulation based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """
        pass

if __name__=='__main__':
    import random
    e = Encoder()

    # create fake data (temporal prediction)
    def mRandom():
        r = random.random()
        if r > 0.7:
            return 1.0
        else:
            return r + 0.4

    def fakeTemporalPrediction():
        prediction = []
        for i in range(1000):
            clip = []
            for j in range(500):
                score = [mRandom() for k in range(10)]
                clip.append(score)
            prediction.append(clip)

        prediction = np.array(prediction)
        print(prediction.shape)

        o = e.encode(prediction)
        print(o)
        for clip in o[:10]:
            print(len(clip[0]), clip[0])


    fakeTemporalPrediction()