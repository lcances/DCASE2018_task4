import os
import argparse

from sklearn.metrics import f1_score

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
from CLR.clr_callback import CyclicLR
from datasetGenerator import DCASE2018
from keras import backend as K
from keras.models import Model

# evaluate
from evaluation_measures import event_based_evaluation
from dcase_util.containers import MetaDataContainer


def modelAlreadyTrained(modelPath: str) -> bool:
    print(modelPath)
    print(modelPath + "_model.json")
    print("MODEL PATH: ", os.path.isfile(modelPath + "_model.json"))
    if not os.path.isfile(modelPath + "_model.json"):
        return False

    if not os.path.isfile(modelPath + "_weight.h5py"):
        return False

    return True

def transferAlreadyDone(modelPath: str) -> bool:
    if not os.path.isfile(modelPath + "_2_model.json"):
        return False

    if not os.path.isfile(modelPath + "_2_weight.h5py"):
        return False

    return True


if __name__ == '__main__':
    # ==================================================================================================================
    #       MANAGE PROGRAM ARGUMENTS
    # ==================================================================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer", help="normalizer [file_MinMax | global_MinMax | file_Mean | global_Mean | file_standard | global_standard | unit")
    parser.add_argument("--output_model", help="basename for save file of the model")
    parser.add_argument("--meta_root", help="Path to the meta directory")
    parser.add_argument("--features_root", help="Path to the features directory")
    parser.add_argument("-uid", help="Use unlabel in domain dataset", action="store_true")
    parser.add_argument("-retrain", help="Force retrain model", action="store_true")
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
    metaRoot = args.meta_root
    featRoot = args.features_root
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
    batch_size = 32
    metrics = ["accuracy", Metrics.f1]
    loss = "binary_crossentropy"
    optimizer = Adam(lr=0.0005)
    print("default lr: ", optimizer.lr)

    completeLogger = CallBacks.CompleteLogger(
        logPath=dirPath,
        validation_data=(dataset.validationDataset["mel"]["input"], dataset.validationDataset["mel"]["output"])
    )

    callbacks = [completeLogger]

    model = Models.crnn_mel64_tr2(dataset)

    # compile & fit model
    if not modelAlreadyTrained(dirPath) or args.retrain:
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

        # save best model (callback history)
        model.set_weights(completeLogger.history[0]["weights"])
        Models.save(dirPath, model)

    else:
        print("Model already build and train, loading ...")
        model = Models.load(dirPath)

    # ==================================================================================================================
    #       Optimize thesholds using the validation dataset
    # ==================================================================================================================
    print("Compute f1 score ...")
    completeLogger.toggleTransfer()
    binarizer = Binarizer()
    encoder = Encoder()

    # save original model and keep track of the best one.
    # Optimize thresholds
    prediction = model.predict(dataset.validationDataset[feat[0]]["input"])
    binarizer.optimize(dataset.validationDataset["mel"]["output"], prediction)
    binPrediction = binarizer.binarize(prediction)
    f1 = f1_score(dataset.validationDataset[feat[0]]["output"], binPrediction, average=None)

    best = {
        "original weight": model.get_weights(), "original f1": f1,
        "transfer weight": model.get_weights(), "transfer f1": f1,
    }

    # ==================================================================================================================
    #       Optimize thesholds using the validation dataset
    # ==================================================================================================================
    if args.uid:
        if not transferAlreadyDone(dirPath) or args.retrain:
            # load the unlabel_in_domain features
            print("Loading the unlabel in domain dataset ...")
            uid_features = dataset.loadUID()

            # Predict the complete unlabel_in_domain dataset and use it to expand the training dataset
            print("Predicting the unlabel in domain dataset ...")
            toPredict = [uid_features[f] for f in feat]
            prediction = model.predict(toPredict)
            binPrediction = binarizer.binarize(prediction)

            print("Expand training dataset and re-training ...")
            dataset.expandWithUID(uid_features, binPrediction)

            # use both weak dataset and unlabel in domain dataset as training dataset
            forTraining = {
                "input": np.concatenate(
                    (dataset.trainingDataset["mel"]["input"], dataset.trainingUidDataset["mel"]["input"])),
                "output": np.concatenate(
                    (dataset.trainingDataset["mel"]["output"], dataset.trainingUidDataset["mel"]["output"]))
            }

            # use the whole weak dataset as validation dataset
            forValidation = {
                "input": np.concatenate(
                    (dataset.trainingDataset["mel"]["input"], dataset.validationDataset["mel"]["input"])),
                "output": np.concatenate(
                    (dataset.trainingDataset["mel"]["output"], dataset.validationDataset["mel"]["output"])),
            }

            # ==================================================================================================================
            #   Train new model with extended dataset (reset the weight)
            # ==================================================================================================================
            trainingIterationPerEpoch = len(forTraining["input"]) / batch_size
            clrCallback = CyclicLR(base_lr = 0.0005, max_lr = 0.003, step_size = trainingIterationPerEpoch, mode='triangular2')

            callbacks.append(clrCallback)

            model2 = Models.crnn_mel64_tr2(dataset)
            model2.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            model2.fit(
                x=forTraining["input"],
                y=forTraining["output"],
                epochs=100,
                validation_data=(
                    forValidation["input"],
                    forValidation["output"]
                ),
                batch_size=128,
                callbacks=callbacks,
                verbose=0
            )

            # save the new model with the _2 extension
            Models.save(dirPath + "_2", model2)

        else:
            print("Transfer already done, loading saved model ...")
            model2 = Models.load(dirPath + "_2")

        # compute f1 score and same (for later comparison)
        print("Compute the final f1 score ...")
        prediction = model2.predict(dataset.validationDataset[feat[0]]["input"])
        binPrediction = binarizer.binarize(prediction)
        f1 = f1_score(dataset.validationDataset[feat[0]]["output"], binPrediction, average=None)
        best["transfer weight"] = model.get_weights()
        best["transfer f1"] = f1

    # ==================================================================================================================
    #   STRONG ANNOTATION stage and evaluation
    # ==================================================================================================================
    if args.uid:
        model2.summary()
        tModel = Model(input=model2.input, output=model2.get_layer("time_distributed_2").output)
    else:
        model.summary()
        tModel = Model(input=model.input, output=model.get_layer("time_distributed_1").output)

    tPrediction = tModel.predict(dataset.testingDataset["mel"]["input"])

    encoder = Encoder()
    segments = encoder.encode(tPrediction, method="threshold")#, smooth="smoothMovingAvg")
    toEvaluate = encoder.parse(segments, dataset.testFileList)

    print("perform evaluation ...")
    with open("toEvaluate.csv", "w") as f:
        f.write("filename\tonset\toffset\tevent_label\n")
        f.write(toEvaluate)

    perso_event_list = MetaDataContainer()
    perso_event_list.load(filename="toEvaluate.csv")

    ref_event_list = MetaDataContainer()
    ref_event_list.load(filename=dataset.meta_test)

    event_based_metric = event_based_evaluation(ref_event_list, perso_event_list)
    print(event_based_metric)

    print("Saving final results in final_results.txt")
    with open("final_results", "w") as f:
        f.write(str(event_based_metric))
