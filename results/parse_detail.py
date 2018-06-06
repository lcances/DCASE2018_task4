import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="Path to file to parse")
parser.add_argument("--name", help="Name of the graph")

args = parser.parse_args()

global classes, colors
classes = ["Alarm_bell_ringing", "Speech", "Dog", "Cat", "Vacuum_cleaner", "Dishes", "Frying", "Electric_shaver_toothbrush", "Blender", "Running_water", "blank"]
colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

def drawLine(high: float) -> tuple:
    x = np.linspace(0, 100, 100)
    y = [high] * x.size

    return y, x

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

def separate_classes(matrix: np.array) -> dict:
    output = {}

    for i in range(len(classes) - 1):
        output[classes[i]] = matrix[:,i+1]

    return output

def getFinalValue(matrix: np.array) -> list:
    return matrix[-1,:][1:]

if __name__=='__main__':
    data_recall = loadFile(args.file + "_recall.csv")
    data_prec = loadFile(args.file + "_precision.csv")
    data_f1 = loadFile(args.file + "_f1.csv")
    hr = getFinalValue(data_recall)
    hp = getFinalValue(data_prec)
    hf = getFinalValue(data_f1)

    plt.figure(figsize=(16, 10))
    plt.title(args.name)
    plt.bar(np.array(range(len(hr))) - 0.2, hr, width= 0.2, tick_label=classes[:-1], zorder=3, color = "C0", label="hr")
    plt.bar(np.array(range(len(hp))), hp, width= 0.2, tick_label=classes[:-1], zorder=3, color = "C1", label="hp")
    plt.bar(np.array(range(len(hf))) + 0.2, hf, width= 0.2, tick_label=classes[:-1], zorder=3, color = "C2", label="hf")
    #plt.xticks(rotation="85")
    plt.grid(zorder=0)
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig(args.file + ".png")
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
