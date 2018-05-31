import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="Path to file to parse")

args = parser.parse_args()

def drawLine(high: float) -> tuple:
    x = np.linspace(0, 100, 100)
    y = [high] * x.size

    return y, x

if __name__=='__main__':
    epoch = []
    with open(args.file, "r") as f:
        lines = f.readlines()

    # get only end of epoch lines
    # ---- get max file by batch
    maxi = 0
    for line in lines:
        try:
            if maxi < int(line[:4]): maxi = int(line[:4])
        except ValueError:
            pass

    for line in lines:
        if line[:9] == "%s/%s" % (maxi, maxi):
            epoch.append(line)

    # get the interresting infomation
    data = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "tra_f1": [], "val_f1": []}
    for e in epoch:
        e = e.split(" ")
        data["loss"].append(float(e[7]))
        data["acc"].append(float(e[10]))
        data["tra_f1"].append(float(e[13]))
        data["val_loss"].append(float(e[16]))
        data["val_acc"].append(float(e[19]))
        data["val_f1"].append(float(e[22][:-1]))

    # display
    plt.subplot(111)
    plt.plot(data["val_loss"], label="val loss", color='C2', linewidth=1)
    plt.plot(data["val_acc"], label="val acc", color='C1', linewidth=1)
    plt.plot(data["val_f1"], label="val f1", color='C0', linewidth=1)

    # print reference line
    plt.plot([0.72]*100, '--C0', label="baseline val f1 = 0.72", linewidth=1)
    plt.title(os.path.basename(args.file).split("-")[0] + " Normalization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.file + ".png")


    """
    fig, ax1 = plt.subplots()
    ax1.plot(data["val_loss"], "b")
    ax1.set_ylabel("val_loss", color='b')
    ax1.set_xlabel('epochs')
    ax1.tick_params('y', color='b')

    ax2 = ax1.twinx()
    ax2.plot(data["val_acc"], "r")
    ax2.set_ylabel("val_acc", color='r')
    ax2.tick_params('y', color='r')

    fig.tight_layout()
    plt.savefig(args.file + ".png")
    """
