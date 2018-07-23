from datasetGenerator import DCASE2018

import keras.utils
from keras.layers import Reshape, BatchNormalization, Activation, MaxPooling2D, Conv2D, Dropout, GRU, Dense
from keras.layers import Input, Bidirectional, TimeDistributed, GlobalAveragePooling1D, Concatenate
from keras.models import Model, model_from_json

def load(dirPath: str) -> Model:
    with open(dirPath + "_model.json", "r") as modelJsonFile:
        model = model_from_json(modelJsonFile.read())
    model.load_weights(dirPath + "_weight.h5py")

    return model

def save(dirPath: str, model: Model, transfer: bool = False):
    # save model ----------
    model_json = model.to_json()
    with open(dirPath + "_model.json", "w") as f:
        f.write(model_json)

    # save weight
    model.save_weights(dirPath + "_weight.h5py")

    if transfer:
        open(dirPath + "_transfer", "w").write("")

def useWGRU(model: Model) -> Model:
    weights = model.get_weights()

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if layers[i].name[:5] == "bidir":
            x = Bidirectional(
                GRU(kernel_initializer='glorot_uniform', recurrent_dropout=0.0, dropout=0.3,
                    units=64, return_sequences=True, temporal_weight=0.25)
            )(x)

        elif layers[i].name[:5] == "time_":
            timeName = layers[i].name
            x = TimeDistributed( Dense(10, activation="sigmoid") )(x)

        else:
            x = layers[i](x)

    newModel = Model(input=layers[0].input, output = x)
    return Model(input=newModel.input, output=newModel.get_layer(timeName).output)

def crnn_mel64_tr2(dataset: DCASE2018) -> Model:
    melInput = Input(dataset.getInputShape("mel"))

    # ---- mel convolution part ----
    mBlock1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(melInput)
    mBlock1 = BatchNormalization()(mBlock1)
    mBlock1 = Activation(activation="relu")(mBlock1)
    mBlock1 = MaxPooling2D(pool_size=(4, 2))(mBlock1)
    mBlock1 = Dropout(0.5)(mBlock1)

    mBlock2 = Conv2D(filters=64, kernel_size=(5, 5), padding="same")(mBlock1)
    mBlock2 = BatchNormalization()(mBlock2)
    mBlock2 = Activation(activation="relu")(mBlock2)
    mBlock2 = MaxPooling2D(pool_size=(4, 1))(mBlock2)
    mBlock2 = Dropout(0.5)(mBlock2)

    mBlock2 = Conv2D(filters=64, kernel_size=(5, 5), padding="same")(mBlock2)
    mBlock2 = BatchNormalization()(mBlock2)
    mBlock2 = Activation(activation="relu")(mBlock2)
    mBlock2 = MaxPooling2D(pool_size=(4, 1))(mBlock2)
    mBlock2 = Dropout(0.5)(mBlock2)

    targetShape = int(mBlock2.shape[1] * mBlock2.shape[2])
    mReshape = Reshape(target_shape=(targetShape, 64))(mBlock2)

    gru = Bidirectional(
        GRU(kernel_initializer='glorot_uniform', recurrent_dropout=0.0, dropout=0.3, units=64, return_sequences=True)
    )(mReshape)

    output = TimeDistributed(
        Dense(dataset.nbClass, activation="sigmoid"),
    )(gru)

    output = GlobalAveragePooling1D()(output)

    model = Model(inputs=[melInput], outputs=output)
    keras.utils.print_summary(model, line_length=100)

    return model

