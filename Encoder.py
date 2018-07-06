import sys
import numpy as np
import copy

from Binarizer import Binarizer
from datasetGenerator import DCASE2018

class Encoder:
    def __init__(self):
        self.frameLength = 0
        self.nbFrame = 0

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

        if method == "threshold": encoder = self.__encodeUsingThreshold
        elif method == "modulation": encoder = self.__encodeUsingModulation
        else:
            sys.exit(1)

        self.nbFrame = temporalPrediction.shape[1]
        self.frameLength = 10 / self.nbFrame
        return encoder(temporalPrediction)

    def __encodeUsingThreshold(self, temporalPrediction: np.array) -> list:
        """ Threshold based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
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

    def __encodeUsingModulation(self, temporalPrediction: np.array):
        """ modulation based localization of the sound event in the clip using the temporal prediction.

        :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
        :return: The result of the system under the form of a strong annotation text where each represent on timed event
        """
        pass

    def parse(self, allSegments: list, testFilesName: list) -> str:
        output = ""

        for clipIndex in range(len(allSegments)):
            clip = allSegments[clipIndex]

            for cls in clip:
                start = 0
                for segment in clip[cls]:
                    if segment[0] == 1.0:
                        output += "%s\t%f\t%f\t%s\n" % (
                            testFilesName[clipIndex][:-4],
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
        if r > 0.4:
            return 0.9
        else:
            return 0.1

    def fakeTemporalPrediction():
        prediction = []
        for i in range(10):
            clip = []
            for j in range(200):
                score = [mRandom() for k in range(10)]
                clip.append(score)
            prediction.append(clip)

        prediction = np.array(prediction)
        print(prediction.shape)

        o = e.encode(prediction)
        t = e.parse(o, prediction[:,0,0])
        print(t)


    fakeTemporalPrediction()