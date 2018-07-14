import os
import argparse

from sklearn.metrics import precision_score, recall_score, f1_score

import random
import numpy.random as npr
import numpy as np
from tensorflow import set_random_seed
from keras.optimizers import Adam


import Models
import Normalizer
import Metrics
from Binarizer import Binarizer
from Encoder import Encoder
import CallBacks
from datasetGenerator import DCASE2018


def modelAlreadyTrained(modelPath: str) -> bool:
    print(modelPath)
    print(modelPath + "_model.json")
    print("MODEL PATH: ", os.path.isfile(modelPath + "_model.json"))
    if not os.path.isfile(modelPath + "_model.json"):
        return False

    if not os.path.isfile(modelPath + "_weight.h5py"):
        return False

    return True

if __name__ == '__main__':
    # ==================================================================================================================
    #       MANAGE PROGRAM ARGUMENTS
    # ==================================================================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer", help="normalizer [file_MinMax | global_MinMax | file_Mean | global_Mean | file_standard | global_standard | unit")
    parser.add_argument("--output_model", help="basename for save file of the model")
    parser.add_argument("-w", help="If set, display the warnigs", action="store_true")
    args = parser.parse_args()

    normalizer = None
    if args.normalizer:
        if args.normalizer == "file_MinMax": normalizer = Normalizer.MinMaxScaler()
        if args.normalizer == "file_Mean": normalizer = Normalizer.MeanScaler()
        if args.normalizer == "file_Standard": normalizer = Normalizer.StandardScaler()
        if args.normalizer == "global_MinMax": normalizer = Normalizer.MinMaxScaler(methods="global")
        if args.normalizer == "global_Mean": normalizer = Normalizer.MeanScaler(methods="global")
        if args.normalizer == "global_Standard": normalizer = Normalizer.StandardScaler(methods="global")
        if args.normalizer == "unit": normalizer = Normalizer.UnitLength()

    if not args.w:
        import warnings
        warnings.filterwarnings("ignore")


    # ==================================================================================================================
    #       INITIALIZE THE PROGRAM AND PREPARE SAVE DIRECTORIES
    # ==================================================================================================================
    # prepare directory
    dirPath = None
    if args.output_model is not None:
        dirPath = args.output_model
        dirName = os.path.dirname(dirPath)
        fileName = os.path.basename(dirPath)

        if not os.path.isdir(dirName):
            os.makedirs(dirName)

    # fix the random seeds
    seed = 1324
    random.seed(seed)
    npr.seed(seed)
    set_random_seed(seed)


    # ==================================================================================================================
    #       PREPARE DATASET
    # ==================================================================================================================
    metaRoot = "../Corpus/DCASE2018/meta"
    featRoot = "../Corpus/DCASE2018/features_2"
    feat = ["mel"]
    dataset = DCASE2018(
        featureRoot=featRoot,
        metaRoot=metaRoot,
        features=feat,
        validationPercent=0.2,
        normalizer=normalizer
    )



    # ==================================================================================================================
    #       Build mode & prepare hyper parameters & train
    #           - if not already done
    # ==================================================================================================================
    # hyperparameters
    epochs = 100
    batch_size = 12
    metrics = ["binary_accuracy", Metrics.precision, Metrics.recall, Metrics.f1]
    loss = "binary_crossentropy"
    optimizer = Adam()
    print("default lr: ", optimizer.lr)
    callbacks = [
        CallBacks.CompleteLogger(logPath=dirPath, validation_data=(dataset.validationDataset["mel"]["input"],
                                                                   dataset.validationDataset["mel"]["output"])
                                 ),
    ]

    model = Models.crnn_mel64_tr2(dataset)

    # compile & fit model
    if not modelAlreadyTrained(dirPath):
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit(
            x=dataset.trainingDataset["mel"]["input"],
            y=dataset.trainingDataset["mel"]["output"],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                dataset.validationDataset["mel"]["input"],
                dataset.validationDataset["mel"]["output"]
            ),
            callbacks=callbacks,
            verbose=0
        )

        # save model ----------
        model_json = model.to_json()
        with open(dirPath + "_model.json", "w") as f:
            f.write(model_json)

        # save weight
        model.save_weights(dirPath + "_weight.h5py")

    else:
        print("Model already build and train, loading ...")
        model = Models.load(dirPath)




    # ==================================================================================================================
    #   Extend original weak dataset by predicting unlabel_in_domain
    # ==================================================================================================================
    print("==== PREDICT UNLABEL_IN_DOMAIN ====")
    featurePath = dict()
    featureFiles = dict()
    features = dict()

    for f in feat:
        featurePath[f] = os.path.join(featRoot, "train", "unlabel_in_domain", f)
        featureFiles[f] = os.listdir(featurePath[f])
        features[f] = []

    # predict the unlabel_in_domain 1000 by 1000 (memory usage limitation 1000 ~= 230 Mo)
    # Each time, retrain model (transfer learning) and save model. Keep only the best model
    # Model are evaluate on their classification score (F1)
    # TODO evaluate also on the localization score
    nbFileToPredict = 1000
    binarizer = Binarizer()
    encoder = Encoder()

    # save original model and keep track of the best one.
    prediction = model.predict(dataset.validationDataset[feat[0]]["input"])
    prediction[prediction > 0.5] = 1        # TODO Change by binarizer
    prediction[prediction < 0.5] = 0        # TODO change by binarizer
    precision = f1_score(dataset.validationDataset[feat[0]]["output"], prediction, average=None)

    best = {
        "original weight": model.get_weights(), "original average f1": precision.mean(),
        "transfer weight": model.get_weights(), "transfer average f1": precision.mean()
    }

    with open(os.path.join(metaRoot, "unlabel_in_domain_semi.csv"), "w") as metaFile:
        for i in range(0, len(featureFiles[feat[0]]) - nbFileToPredict, nbFileToPredict):
            toLoad = dict()

            for f in feat:
                toLoad[f] = featureFiles[f][i:i+nbFileToPredict]

            # retrieve the features (already extracted)
            featureLoaded = dict()
            for f in feat:
                featureLoaded[f] = []

            for j in range(nbFileToPredict):
                for f in feat:
                    feature = np.load(os.path.join(featurePath[f], toLoad[f][j]))

                    # pre processing
                    feature = np.expand_dims(feature, axis=-1)

                    featureLoaded[f].append(feature)

            # predict the <nbFileToPredict> files loaded in memory
            toPredictList = [featureLoaded[f] for f in feat]
            prediction = model.predict(toPredictList)
            binPrediction = binarizer.binarize(prediction)
            binPredictionCls = encoder.binToClass(binPrediction)

            # write the new metadata for unlabel_in_domain newly annotated
            unlabelInDomainWeakMeta = ""
            labels = []
            for k in range(nbFileToPredict):
                fileName = toLoad[feat[0]][k]
                unlabelInDomainWeakMeta += "%s %s\n" % (fileName, binPredictionCls[k])
                labels.append(binPredictionCls[k].split(","))

            # save the new metadata file
            print("Saving results %s to %s" % (i, i+nbFileToPredict))
            metaFile.write(unlabelInDomainWeakMeta)

            # Retrain the model and check if better than previous one
            newOutput = []
            for label in labels:

                output = [0] * 10
                for l in label:
                    if l != "":
                        output[DCASE2018.class_correspondance[l]] = 1
                newOutput.append(output)
            newOutput = np.array(newOutput)

            optimizer.lr = 0.00001  # 100 times smaller
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            model.fit(
                x=[np.array(featureLoaded[feat[0]])],
                y=newOutput,
                epochs=80,
                validation_data=(
                    dataset.validationDataset["mel"]["input"],
                    dataset.validationDataset["mel"]["output"]
                ),
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )

            print("PREDICTION\n\n\n")
            prediction = model.predict(dataset.validationDataset[feat[0]]["input"])
            prediction[prediction > 0.5] = 1        # TODO use binarizer
            prediction[prediction < 0.5] = 0        # TODO use binarizer
            print("\n\n\nSCORE\n\n\n")
            precision = f1_score(dataset.validationDataset[feat[0]]["output"], prediction, average=None)
            print("original: %.5f <--> %.5f transfer" % (best["original average f1"], precision.mean()) )

            # save model if better
            if precision.mean() > best["transfer average f1"]:
                best["transfer average f1"] = precision.mean()
                best["transfer weight"] = model.get_weights()

    # Overwrite the saved model if the transfer one is better
    if best["original average f1"] < best["transfer average f1"]:
        Models.save(dirPath, transfer=True)

    # Otherwise go back to the original model
    else:
        model.set_weights(best["original weight"])

    print("TRANSFER LEARNING FINISH !!!!!!")
    print("F1 score")
    print("original: ", best["original average f1"])
    print("tranfer: ", best["transfer average f1"])

