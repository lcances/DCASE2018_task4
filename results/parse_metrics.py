import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

class MetricParser:

    @staticmethod
    def drawLine(high: float) -> tuple:
        x = np.linspace(0, 100, 100)
        y = [high] * x.size

        return y, x

    @staticmethod
    def loadFile(path: str):
        output = []
        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines[2:]:
            detail = line.split(",")

            # reformat last column
            detail[-1] = detail[-1][:5]
            output.append(detail)

        return np.array(output, dtype=float)

    @staticmethod
    def separate_metrics(matrix: np.array):
        return {
            "epoch": matrix[:,0],
            "loss": matrix[:,2],
            "binary acc": matrix[:,3],
            "precision": matrix[:,4],
            "recall": matrix[:,5],
            "f1": matrix[:,6],
            "val loss": matrix[:,7],
            "val binary acc": matrix[:,8],
            "val precision": matrix[:,9],
            "val recall": matrix[:,10],
            "val f1": matrix[:,11],
            "max val f1": max(matrix[:,11]),
            "max recall": max(matrix[:,10]),
            "max precision": max(matrix[:,9]),
            "max val binary acc": max(matrix[:,8])
        }

    @staticmethod
    def findbestEpoch(f1: list):
        cpt = 0
        maxi = 0
        maxiEpoch = 0
        for fvalue in f1:
            cpt += 1
            if fvalue > maxi:
                maxi = fvalue
                maxiEpoch = cpt
        return maxiEpoch

    @staticmethod
    def parseAndSave(path: str, name: str) -> int:
        data = MetricParser.loadFile(path)
        data = MetricParser.separate_metrics(data)

        # display
        plt.figure(figsize=(16, 10))
        plt.subplot(111)
        plt.plot(data["val loss"], label="val loss", color='C2', linewidth=1)
        plt.plot(data["loss"], label="tra loss", color='C2', linewidth=1, alpha=0.5)
        plt.plot(data["val binary acc"], label="val acc (%s)" % data["max val binary acc"], color='C1', linewidth=1)
        plt.plot(data["binary acc"], label="tra acc", color='C1', linewidth=1, alpha=0.5)
        plt.plot(data["val f1"], label="val f1 (%s)" % data["max val f1"], color='C0', linewidth=1)
        plt.plot(data["f1"], label="tra f1", color='C0', linewidth=1, alpha=0.5)

        # print reference line
        plt.plot([0.72]*100, '--C0', label="baseline val f1 = 0.72", linewidth=1)
        plt.title(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(name + ".png")

        return MetricParser.findbestEpoch(data["val f1"])

