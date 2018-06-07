import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

class DetailParser:
    CLASSES = ["Alarm_bell_ringing", "Speech", "Dog", "Cat", "Vacuum_cleaner", "Dishes", "Frying", "Electric_shaver_toothbrush", "Blender", "Running_water", "blank"]

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
    def separate_classes(matrix: np.array) -> dict:
        output = {}

        for i in range(len(DetailParser.CLASSES) - 1):
            output[DetailParser.CLASSES[i]] = matrix[:,i+1]

        return output

    @staticmethod
    def getFinalValue(matrix: np.array, epoch: int) -> list:
        if epoch >= matrix.shape[0]:
            epoch = matrix.shape[0] - 1

        return matrix[int(epoch),:][1:]

    @staticmethod
    def parseAndSave(path: str, name: str, epoch: int):
        data_recall = DetailParser.loadFile(path + "_recall.csv")
        data_prec = DetailParser.loadFile(path + "_precision.csv")
        data_f1 = DetailParser.loadFile(path + "_f1.csv")
        hr = DetailParser.getFinalValue(data_recall, epoch)
        hp = DetailParser.getFinalValue(data_prec, epoch)
        hf = DetailParser.getFinalValue(data_f1, epoch)

        plt.figure(figsize=(16, 10))
        plt.title(name)
        plt.bar(np.array(range(len(hr))) - 0.2, hr, width= 0.2, tick_label=DetailParser.CLASSES, zorder=3, color = "C0", label="recall")
        plt.bar(np.array(range(len(hp))), hp, width= 0.2, tick_label=DetailParser.CLASSES, zorder=3, color = "C1", label="precision")
        plt.bar(np.array(range(len(hf))) + 0.2, hf, width= 0.2, tick_label=DetailParser.CLASSES, zorder=3, color = "C2", label="f1")
        #plt.xticks(rotation="85")
        plt.grid(zorder=0)
        plt.tight_layout()
        plt.legend()
        plt.savefig(name + ".png")

    """
    # display
    print("display")
    plt.figure(figsize=(16, 10))
    plt.subplot(111)

    colorPicker = 0
    for key in data:
        plt.plot(data[key], label=key, color=colors[colorPicker], linewidth=1)
        colorPicker  = (colorPicker + 1) % len(colors)


    # print reference line
    plt.title(args.name)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(args.file + ".png")
    """
