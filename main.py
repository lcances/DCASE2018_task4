import os

import keras.utils
from keras.layers import Reshape, BatchNormalization, Activation, MaxPooling2D, Conv2D, Dropout, GRU, Dense, \
    Input, Bidirectional, TimeDistributed, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Nadam

import Normalizer
import Metrics
import CallBacks
from datasetGenerator import DCASE2018

if __name__ == '__main__':
    # deactivate warning (TMP)
    import warnings
    warnings.filterwarnings("ignore")

    # ARGUMENT PARSER ====
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer", help="normalizer [file_MinMax | global_MinMax | file_Mean | global_Mean | file_standard | global_standard | unit")
    parser.add_argument("--output_model", help="basename for save file of the model")
    args = parser.parse_args()

    # PROCESS ARGUMENTS ====
    normalizer = None
    if args.normalizer:
        if args.normalizer == "file_MinMax": normalizer = Normalizer.File_MinMaxNormalization
        if args.normalizer == "global_MinMax": normalizer = Normalizer.Global_MinMaxNormalization
        if args.normalizer == "file_Mean": normalizer = Normalizer.File_MeanNormalization
        if args.normalizer == "global_Mean": normalizer = Normalizer.Global_MeanNormalization
        if args.normalizer == "file_Standard": normalizer = Normalizer.File_Standardization
        if args.normalizer == "global_Standard": normalizer = Normalizer.Global_Standardization
        if args.normalizer == "unit": normalizer = Normalizer.UnitLength

    # Prepare the save directory (if needed)
    dirPath = None
    if args.output_model is not None:
        dirPath = args.output_model
        dirName = os.path.dirname(dirPath)
        fileName = os.path.basename(dirPath)

        if not os.path.isdir(dirName):
            print("File doesn't exist, creating it")
            os.makedirs(dirName)

    # GENERATE DATASET ====
    dataset = DCASE2018(
        meta_train_weak="meta/weak.csv",
        meta_train_unlabelOutDomain="meta/unlabel_out_of_domain.csv",
        feat_train_weak="features_2/train/weak/mel",
        feat_train_unlabelOutDomain="features_2/train/unlabel_out_of_domain/mel",
        #feat_train_weak="C:/Users/leo/Documents/Cours/M2/MasterThesis/Python/DCASE2018/features/features/train/weak/mel",
        normalizer=normalizer
    )
    print(dataset)

    # MODEL HYPERPARAMETERS ====
    epochs = 100
    batch_size = 4
    metrics = ["binary_accuracy", Metrics.precision, Metrics.recall, Metrics.f1]
    loss = "binary_crossentropy"
    optimizer = "adam"
    callbacks = [
        CallBacks.CompleteLogger(logPath=dirPath, validation_data=(dataset.validationDataset["input"],
                                                                   dataset.validationDataset["output"])
                                 ),
    ]

    # ==================================================================================================================
    #   Creating the model
    # ==================================================================================================================
    kInput = Input(dataset.getInputShape())

    block1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(kInput)
    block1 = BatchNormalization()(block1)
    block1 = Activation(activation="relu")(block1)
    block1 = MaxPooling2D(pool_size=(1, 4))(block1)
    block1 = Dropout(0.3)(block1)

    block2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation(activation="relu")(block2)
    block2 = MaxPooling2D(pool_size=(1, 4))(block2)
    block2 = Dropout(0.3)(block2)

    block3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation(activation="relu")(block3)
    block3 = MaxPooling2D(pool_size=(1, 4))(block3)
    block3 = Dropout(0.3)(block3)

    targetShape = dataset.originalShape[0]
    reshape = Reshape(target_shape=(targetShape, 64))(block3)

    gru = Bidirectional(
        GRU(kernel_initializer='glorot_uniform', recurrent_dropout=0.0, dropout=0.3, units=64, return_sequences=True)
    )(reshape)
    print(gru.shape)

    output = TimeDistributed(
        Dense(dataset.nbClass, activation="sigmoid"),
    )(gru)
    print(output.shape)

    output = GlobalAveragePooling1D()(output)

    model = Model(inputs=kInput, outputs=output)
    keras.utils.print_summary(model, line_length=100)

    # compile & fit model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(
        x=dataset.trainingDataset["input"],
        y=dataset.trainingDataset["output"],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            dataset.validationDataset["input"],
            dataset.validationDataset["output"]
        ),
        callbacks=callbacks,
        verbose=0
    )

    # save json
    model_json = model.to_json()
    with open(dirPath + "_model.json", "w") as f:
        f.write(model_json)

    # save weight
    model.save_weights(dirPath + "_weight.h5py")


