import keras.utils
from keras.layers import Reshape, BatchNormalization, Activation, MaxPooling2D, Conv2D, Dropout, GRU, Dense, \
    Input, Bidirectional, TimeDistributed, GlobalAveragePooling1D
from keras.models import Model
from keras import backend as K

import Normalizer
import dcase_util
from datasetGenerator import DCASE2018


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer", help="normalizer [file_MinMax | global_MinMax | file_Mean | global_Mean |"
                                             " file_standard | global_standard")

    args = parser.parse_args()

    normalizer = None
    if args.normalizer:
        if args.normalizer == "file_MinMax": normalizer = Normalizer.File_MinMaxNormalization
        if args.normalizer == "global_MinMax": normalizer = Normalizer.Global_MinMaxNormalization
        if args.normalizer == "file_Mean": normalizer = Normalizer.File_MeanNormalization
        if args.normalizer == "global_Mean": normalizer = Normalizer.Global_MeanNormalization
        if args.normalizer == "file_Standard": normalizer = Normalizer.File_Standardization
        if args.normalizer == "global_Standard": normalizer = Normalizer.Global_Stadardization

    dataset = DCASE2018(
        meta_train_weak="meta/weak.csv",
        feat_train_weak="/homeLocal/eriador/Documents/DCASE2018/task4/features/train/weak/mel",
        normalizer=normalizer
    )

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
    # block3 ndim = 4

    targetShape = dataset.originalShape[0]
    reshape = Reshape(target_shape=(targetShape, 64))(block3)
    # reshape ndim = 3

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

    # model hyper parameters
    epochs = 100
    metrics = "binary_accuracy"
    loss = "binary_crossentropy"
    ProgressLoggerCallbackParam = {
        "external_metric_labels": {
            "val_macro_f_measure": "val_macro_f_measure",
            "tra_macro_f_measure": "tra_macro_f_measure"
        },
        "manual_update": True,
        "processing_interval": 1
    }
    StopperCallbacksParam = {
        "manual_update": True,
        "monitor": "val_macro_f_measure",
        "initial_delay": 5,
        "min_delta": 0.01,  # be careful, means 1%,
        "patience": 15,
        "external_metric_labels": {
            "val_macro_f_measure": "val_macro_f_measure",
            "tra_macro_f_measure": "tra_macro_f_measure"
        }
    }
    StasherCallbacksParam = {
        "manual_update": True,
        "monitor": "val_macro_f_measure",
        "initial_delay": 5,
        "external_metric_labels": {
            "val_macro_f_measure": "val_macro_f_measure",
            "tra_macro_f_measure": "tra_macro_f_measure"
        }
    }

    # f1 metrics

    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


    # compile model
    model.compile(loss=loss,
                  optimizer="adam",
                  metrics=[metrics, f1])


    # callbacks
    ProgressLoggerCallback = dcase_util.keras.ProgressLoggerCallback(
        epochs=epochs,
        metric=metrics,
        loss=loss,
        output_type='logging',
        **ProgressLoggerCallbackParam
    )

    StopperCallbacks = dcase_util.keras.StopperCallback(
        epochs=epochs,
        **StopperCallbacksParam

    )

    StasherCallbacks = dcase_util.keras.StasherCallback(
        epochs=epochs,
        **StasherCallbacksParam
    )

    # fit
    model.fit(
        x=dataset.trainingDataset["input"],
        y=dataset.trainingDataset["output"],
        epochs=200,
        batch_size=8,
        validation_data=(
            dataset.validationDataset["input"],
            dataset.validationDataset["output"]
        ),
        callbacks=[ProgressLoggerCallback, StopperCallbacks, StasherCallbacks]
    )
