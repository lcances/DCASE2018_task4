from datasetGenerator import DCASE2018

import keras.utils
from keras.layers import Reshape, BatchNormalization, Activation, MaxPooling2D, Conv2D, Dropout, GRU, Dense, \
        Input, Bidirectional, TimeDistributed, GlobalAveragePooling1D, Concatenate, GRUCell, SpatialDropout2D, \
        Flatten, Multiply

from keras.models import Model, model_from_json
from keras import backend as K
from keras import regularizers

class CustomGRUCell(GRUCell):

    def __init__(self, units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None,
                 recurrent_constraint=None, bias_constraint=None, dropout=0., recurrent_dropout=0., implementation=1,
                 reset_after=False, temporal_weight: float = 0.5, **kwargs):

        self.temporal_weight = temporal_weight

        super().__init__(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer,
                         bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer,
                         kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout,
                         implementation, reset_after, **kwargs)

        print("Temporal weight : ", self.temporal_weight)

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory

        # if 0 < self.dropout < 1 and self._dropout_mask is None:
        #     self._dropout_mask = _generate_dropout_mask(
        #         K.ones_like(inputs),
        #         self.dropout,
        #         training=training,
        #         count=3)
        # if (0 < self.recurrent_dropout < 1 and
        #         self._recurrent_dropout_mask is None):
        #     self._recurrent_dropout_mask = _generate_dropout_mask(
        #         K.ones_like(h_tm1),
        #         self.recurrent_dropout,
        #         training=training,
        #         count=3)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            x_z = K.dot(inputs_z, self.kernel_z)
            x_r = K.dot(inputs_r, self.kernel_r)
            x_h = K.dot(inputs_h, self.kernel_h)
            if self.use_bias:
                x_z = K.bias_add(x_z, self.input_bias_z)
                x_r = K.bias_add(x_r, self.input_bias_r)
                x_h = K.bias_add(x_h, self.input_bias_h)






            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * self.temporal_weight  # rec_dp_mask[0]
                h_tm1_r = h_tm1 * self.temporal_weight  # rec_dp_mask[1]
                h_tm1_h = h_tm1 * self.temporal_weight  # rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1 * self.temporal_weight
                h_tm1_r = h_tm1 * self.temporal_weight
                h_tm1_h = h_tm1 * self.temporal_weight






            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel_z)
            recurrent_r = K.dot(h_tm1_r, self.recurrent_kernel_r)
            if self.reset_after and self.use_bias:
                recurrent_z = K.bias_add(recurrent_z, self.recurrent_bias_z)
                recurrent_r = K.bias_add(recurrent_r, self.recurrent_bias_r)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, self.recurrent_bias_h)
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(r * h_tm1_h, self.recurrent_kernel_h)

            hh = self.activation(x_h + recurrent_h)
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = K.dot(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = K.bias_add(matrix_x, self.input_bias)
            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            x_h = matrix_x[:, 2 * self.units:]

            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, self.recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = K.dot(h_tm1,
                                     self.recurrent_kernel[:, :2 * self.units])

            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * matrix_inner[:, 2 * self.units:]
            else:
                recurrent_h = K.dot(r * h_tm1,
                                    self.recurrent_kernel[:, 2 * self.units:])

            hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h]


class CustomGRU(GRU):

    def __init__(self, units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.,
                 recurrent_dropout=0., implementation=1, return_sequences=False, return_state=False, go_backwards=False,
                 stateful=False, unroll=False, reset_after=False, temporal_weight: float = 0.5, **kwargs):
        """
        super().__init__(units, activation=activation, recurrent_activation=recurrent_activation,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
                         bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                         recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint,
                         dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation,
                         return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards,
                         stateful=stateful, unroll=unroll, reset_after=reset_after, **kwargs)
        """

        self.temporal_weight = temporal_weight

        cell = CustomGRUCell(units,
                             activation=activation,
                             recurrent_activation=recurrent_activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             recurrent_initializer=recurrent_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             recurrent_regularizer=recurrent_regularizer,
                             bias_regularizer=bias_regularizer,
                             kernel_constraint=kernel_constraint,
                             recurrent_constraint=recurrent_constraint,
                             bias_constraint=bias_constraint,
                             dropout=dropout,
                             recurrent_dropout=recurrent_dropout,
                             implementation=implementation,
                             reset_after=reset_after,
                             temporal_weight=temporal_weight)

        super(GRU, self).__init__(cell,
                                  return_sequences=return_sequences,
                                  return_state=return_state,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def get_config(self):
        config = super().get_config()
        config["temporal_weight"] = self.temporal_weight
        return config

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(inputs, mask, True, initial_state)


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

def useWGRU(modelPath: str) -> Model:
    with open(modelPath + "_model.json") as modelJsonFile:
        model = model_from_json(modelJsonFile.read())
        print("model loaded")

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if layers[i].name[:5] == "bidir":
            x = Bidirectional(
                CustomGRU(units = 64, kernel_initializer='glorot_uniform', recurrent_dropout=0.8,
                    dropout=0.0, return_sequences=True, temporal_weight=0.25), name="custom_bi")(x)

        elif layers[i].name[:5] == "time_":
            timeName = layers[i].name
            print("name ::::::: ", timeName)
            x = TimeDistributed( Dense(10, activation="sigmoid") )(x)

        else:
            x = layers[i](x)
        print(x)

    newModel = Model(input=layers[0].input, output = x)
    newModel.load_weights(modelPath +"_weight.h5py")

    return Model(input=newModel.input, output=newModel.get_layer("time_distributed_1").output)


def crnn_mel64_tr2(dataset: DCASE2018) -> Model:
    melInput = Input(dataset.getInputShape("mel"))

    # ---- mel convolution part ----
    mBlock1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(melInput)
    mBlock1 = BatchNormalization()(mBlock1)
    mBlock1 = Activation(activation="relu")(mBlock1)
    mBlock1 = MaxPooling2D(pool_size=(4, 2))(mBlock1)
    #mBlock1 = SpatialDropout2D(0.15, data_format="channels_last")(mBlock1)
    mBlock1 = Dropout(0.4)(mBlock1)

    mBlock2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(mBlock1)
    mBlock2 = BatchNormalization()(mBlock2)
    mBlock2 = Activation(activation="relu")(mBlock2)
    mBlock2 = MaxPooling2D(pool_size=(4, 1))(mBlock2)
    #mBlock2 = SpatialDropout2D(0.15, data_format="channels_last")(mBlock2)
    mBlock2 = Dropout(0.4)(mBlock2)

    mBlock2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(mBlock2)
    mBlock2 = BatchNormalization()(mBlock2)
    mBlock2 = Activation(activation="relu")(mBlock2)
    mBlock2 = MaxPooling2D(pool_size=(4, 1))(mBlock2)
    #mBlock2 = SpatialDropout2D(0.15, data_format="channels_last")(mBlock2)
    mBlock2 = Dropout(0.4)(mBlock2)

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

def cnn_att(dataset: DCASE2018) -> Model:
    melInput = Input(dataset.getInputShape("mel"))

    # ---- mel convolution part ----
    conv = melInput
    # first conv -> time reduction / 2
    conv = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(activation="relu")(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    conv = SpatialDropout2D(0.15, data_format="channels_last")(conv)

    filters = [64, 64, 64, 64]
    for fSize in filters:
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation(activation="relu")(conv)
        conv = MaxPooling2D(pool_size=(2, 1))(conv)
        conv = SpatialDropout2D(0.15, data_format="channels_last")(conv)
        #conv = Dropout(0.5)(conv)

    # last conv -> normal + attention layer
    link = conv

    conv = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(activation="relu")(conv)
    conv = MaxPooling2D(pool_size=(2, 1))(conv)
    conv = SpatialDropout2D(0.15, data_format="channels_last")(conv)

    att = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(link)
    att = BatchNormalization()(att)
    att = Activation(activation="sigmoid")(att)
    att = MaxPooling2D(pool_size=(2, 1))(att)
    att = SpatialDropout2D(0.15, data_format="channels_last")(att)

    mult = Multiply()([conv, att])

    dense = Flatten()(mult)

    dense = Dense(1500, activation="relu")(dense)
    dense = Dense(796, activation="relu")(dense)
    dense = Dense(256, activation="relu")(dense)
    dense = Dense(10, activation="sigmoid")(dense)

    model = Model(inputs=[melInput], outputs=dense)
    keras.utils.print_summary(model, line_length=100)

    return model

def dense_crnn_mel64_tr2(dataset: DCASE2018) -> Model:
    def convBlock(entry):
        conv = BatchNormalization()(entry)
        conv = Activation(activation="relu")(conv)
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv)
        #conv = Dropout(0.5)(conv)
        conv = SpatialDropout2D(0.30, data_format="channels_last")(conv)

        return conv

    def denseBlock(entry, size):
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(entry)

        links = [conv]
        for i in range(size-1):
            conv = convBlock(entry)
            links.append(conv)
            x = Concatenate(axis=-1)(links)

        return x

    denseBlockSize = 5
    melInput = Input(dataset.getInputShape("mel"))

    # ---- mel convolution part ----
    conv = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(melInput)

    # denseBlock 1
    conv = denseBlock(conv, denseBlockSize)

    conv = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(conv)
    conv = MaxPooling2D(pool_size=(4, 2))(conv)

    # denseBlock 2
    conv = denseBlock(conv, denseBlockSize)

    conv = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(conv)
    conv = MaxPooling2D(pool_size=(4, 1))(conv)

    # denseBlock 3
    conv = denseBlock(conv, denseBlockSize)

    conv = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(conv)
    conv = MaxPooling2D(pool_size=(4, 1))(conv)

    print(conv)
    targetShape = int(conv.shape[1] * conv.shape[2])
    mReshape = Reshape(target_shape=(targetShape, 64))(conv)

    gru = Bidirectional(
        GRU(kernel_initializer='glorot_uniform', recurrent_dropout=0.0, dropout=0.3, units=64, return_sequences=True)
    )(mReshape)

    output = TimeDistributed(
        Dense(100, activation="relu"),
    )(gru)

    output = TimeDistributed(
        Dense(dataset.nbClass, activation="sigmoid"),
    )(output)

    output = GlobalAveragePooling1D()(output)

    model = Model(inputs=[melInput], outputs=output)

    return model

if __name__=='__main__':
    dataset = DCASE2018(
        featureRoot="/baie/corpus/DCASE2018/task4/FEATURES",
        metaRoot="/baie/corpus/DCASE2018/task4/metadata",
        features=["mel"],
        validationPercent=0.2,
        normalizer=None
    )

    model = dense_crnn_mel64_tr2(dataset)
    keras.utils.print_summary(model, line_length=100)

