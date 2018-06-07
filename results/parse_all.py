import argparse
import os
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", help="dir where the data are")

args = parser.parse_args()

listFiles = os.listdir(args.d)

# remove element that are can't be parse
extensions = ["csv"]
toParse = [i for i in listFiles if i.split(".")[-1] in extensions]

# groups the files
def addToDict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

files = {}
for f in toParse:
    detail = f.split("_")
    addToDict(files, detail[0], detail[1])

# parse all files
from parse_metrics import MetricParser
from parse_detail import DetailParser

with tqdm.tqdm(total=len(files), unit="Files") as progress:
    for f in files:
        if args.d is not None:
            path = os.path.join(args.d, f)
        else:
            path = os.path.join(".", f)

        e = MetricParser.parseAndSave(path+"_metrics.csv", f + "_metrics")
        DetailParser.parseAndSave(path, f, e)

        progress.update()

