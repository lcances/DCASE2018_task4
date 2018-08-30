import os
import argparse

from sklearn.metrics import f1_score

import random
import numpy.random as npr
import numpy as np
from tensorflow import set_random_seed

from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import Models
import Normalizer
import Metrics
from Binarizer import Binarizer
from Encoder import Encoder
import CallBacks
from datasetGenerator import DCASE2018

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
    parser.add_argument("--normalizer", type=str, help="normalizer [file_MinMax | global_MinMax | file_Mean | global_Mean | file_standard | global_standard | unit")
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
        if args.normalizer == "unit": normalizer = Normalizer.UnitScaler()

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
        feature_root=featRoot,
        meta_root=metaRoot,
        features=feat,
        validation_percent=0.2,
        normalizer=normalizer
    )

    # ==================================================================================================================
    #       Build mode & prepare hyper parameters & train
    #           - if not already done
    # ==================================================================================================================
    # hyperparameters
    epochs = 200
    batch_size = 8
    metrics = ["accuracy", Metrics.f1]
    loss = "binary_crossentropy"
    optimizer = Adam(lr=0.0005)
    print("default lr: ", optimizer.lr)

    completeLogger = CallBacks.CompleteLogger(
        logPath=dirPath,
        validation_data=(dataset.validation_dataset["mel"]["input"], dataset.validation_dataset["mel"]["output"]),
        fallback = True, fallBackThreshold = 3, stopAt = 100
    )
    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True)
    reduce_lr = ReduceLROnPlateau(factor=0.15, patience=5, min_lr=0.000005)

    callbacks = [completeLogger, early_stopping, model_checkpoint, reduce_lr]

    # compile & fit model
    if not modelAlreadyTrained(dirPath) or args.retrain:
        model = Models.crnn_mel64_tr2(dataset)
        #model = Models.dense_crnn_mel64_tr2(dataset)

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit(
            x=dataset.training_dataset["mel"]["input"],
            y=dataset.training_dataset["mel"]["output"],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                dataset.validation_dataset["mel"]["input"],
                dataset.validation_dataset["mel"]["output"]
            ),
            callbacks=callbacks,
            verbose=0
        )

        # save best model (callback history)
        print("Best model:")
        print("\t epoch: %s" % completeLogger.sortedHistory[-1]["epoch"])
        print("\t mean f1: %s" % completeLogger.sortedHistory[-1]["average f1"])
        model.set_weights(completeLogger.sortedHistory[-1]["weights"])
        #model = load_model("./keras.model")

        Models.save(dirPath, model)

    else:
        print("Model already build and train, loading ...")
        model = Models.load(dirPath)

    # ==================================================================================================================
    #       Optimize thesholds using the validation dataset
    # ==================================================================================================================
    print("Compute f1 score on evaluation dataset")
    completeLogger.toggleTransfer()
    binarizer = Binarizer()
    encoder = Encoder()

    # save original model and keep track of the best one.
    # Optimize thresholds
    prediction = model.predict(dataset.validation_dataset[feat[0]]["input"])
    binPrediction = binarizer.binarize(prediction)
    f10 = f1_score(dataset.validation_dataset[feat[0]]["output"], binPrediction, average=None)

    binarizer.optimize(dataset.validation_dataset["mel"]["output"], prediction)
    binPrediction = binarizer.binarize(prediction)
    f1 = f1_score(dataset.validation_dataset[feat[0]]["output"], binPrediction, average=None)

    best = {
        "original weight": model.get_weights(), "original f1": f10,
        "transfer weight": model.get_weights(), "transfer f1": f1,
    }

    print("original f1", best["original f1"].mean())
    print(best["original f1"])
    print("")
    print("optimized f1", best["transfer f1"].mean())
    print(best["transfer f1"])

    # ==================================================================================================================
    #       Optimize thesholds using the validation dataset
    # ==================================================================================================================
    if args.uid:
        if not transferAlreadyDone(dirPath) or args.retrain:
            # load the unlabel_in_domain features
            print("Loading the unlabel in domain dataset ...")
            uid_features = dataset.load_uid()

            # Predict the complete unlabel_in_domain dataset and use it to expand the training dataset
            print("Predicting the unlabel in domain dataset ...")
            toPredict = [uid_features[f] for f in feat]
            prediction = model.predict(toPredict)
            binPrediction = binarizer.binarize(prediction)

            print("Expand training dataset and re-training ...")
            dataset.expand_with_uid(uid_features, binPrediction)

            # use both weak dataset and unlabel in domain dataset as training dataset
            forTraining = {
                "input": np.concatenate(
                    (dataset.training_dataset["mel"]["input"], dataset.training_uid_dataset["mel"]["input"])),
                "output": np.concatenate(
                    (dataset.training_dataset["mel"]["output"], dataset.training_uid_dataset["mel"]["output"]))
            }

            # use the whole weak dataset as validation dataset
            forValidation = {
                "input": np.concatenate(
                    (dataset.training_dataset["mel"]["input"], dataset.validation_dataset["mel"]["input"])),
                "output": np.concatenate(
                    (dataset.training_dataset["mel"]["output"], dataset.validation_dataset["mel"]["output"])),
            }

            # ==================================================================================================================
            #   Train new model with extended dataset (reset the weight)
            # ==================================================================================================================
            trainingIterationPerEpoch = len(forTraining["input"]) / batch_size

            #m odel2 = Models.dense_crnn_mel64_tr2(dataset)
            model_2 = Models.crnn_mel64_tr2(dataset)

            model_2.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            model_2.fit(
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
            #model2 = load_model("./keras.model")
            model_2.set_weights(completeLogger.sortedHistory[-1]["weights"])
            Models.save(dirPath + "_2", model_2)

        else:
            print("Transfer already done, loading saved model ...")
            model_2 = Models.load(dirPath + "_2")

        # compute f1 score and same (for later comparison)
        print("Compute the final f1 score ...")
        prediction = model_2.predict(dataset.validation_dataset[feat[0]]["input"])
        binPrediction = binarizer.binarize(prediction)
        f1 = f1_score(dataset.validation_dataset[feat[0]]["output"], binPrediction, average=None)
        best["transfer weight"] = model.get_weights()
        best["transfer f1"] = f1

    # ==================================================================================================================
    #   STRONG ANNOTATION stage and evaluation
    # ==================================================================================================================
    if args.uid:
        model_2.summary()
        g_model = model_2
        t_model = Model(input=model_2.input, output=model_2.get_layer("time_distributed_1").output)
        tw_model = Models.use_wgru(dirPath + "_2")

    else:
        model.summary()
        g_model = model
        t_model = Model(input=model.input, output=model.get_layer("time_distributed_1").output)
        tw_model = Models.use_wgru(dirPath)

    # global prediction
    # gPrediction = gModel.predict(dataset.testingDataset["mel"]["input"])
    # gbPrediction = binarizer.binarize(gPrediction)

    # temporal prediction using both WGRU and GRU
    t_prediction = t_model.predict(dataset.testing_dataset["mel"]["input"])
    tw_prediction = tw_model.predict(dataset.testing_dataset["mel"]["input"])
    nb_frame = t_prediction.shape[1]
    print(t_prediction.shape)

    # mix the prediction giving the globals prediction
    wgru_cls = [0, 2, 1]               # "impulse" event well detected by the WGRU
    gru_cls = [8, 3, 5, 7, 6, 9, 4]   # "stationary" event well detected by the GRU

    final_t_prediction = []
    for i in range(len(t_prediction)):

        curves = np.array([[0] * dataset.nbClass for _ in range(nb_frame)], dtype=np.float32)

        # use the WGRU temporal
        for c in wgru_cls:
            curves[:,c] = tw_prediction[i][:, c] # use the wgru temporal prediction

        for c in gru_cls:
            curves[:,c] = t_prediction[i][:, c]  # use the classic gru temporal prediction

        final_t_prediction.append(curves)
    final_t_prediction = np.array(final_t_prediction)
    print(final_t_prediction.shape)

    encoder = Encoder()
    segments = encoder.encode(final_t_prediction, method="threshold", smooth="smoothMovingAvg")
    toEvaluate = encoder.parse(segments, dataset.test_file_list)

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
    with open("final_results.txt", "w") as f:
        f.write(str(event_based_metric))
