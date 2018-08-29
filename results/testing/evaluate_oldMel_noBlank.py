
# coding: utf-8

# # Evaluation model OLD MEL NO BLANK

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '')

# Load the required modules
import sys
sys.path.append("../..")
from datasetGenerator import DCASE2018
from Encoder import Encoder
from Binarizer import Binarizer

from keras.layers import GRU, Bidirectional, Layer, TimeDistributed, Dense, GRUCell
from keras.models import model_from_json, Model
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from subprocess import call

# evaluate
from evaluation_measures import event_based_evaluation
from dcase_util.containers import MetaDataContainer

import librosa.display
import librosa
import numpy as np
import os
import copy


# # Load the dataset

# In[2]:


# Load the test data create the test dataset
# load the file list
featTestPath = "/homeLocal/eriador/Documents/Corpus/DCASE2018/features_2/test/mel"
featTestList = os.listdir(featTestPath)

# load the meta data ----
metaPath = "/homeLocal/eriador/Documents/Corpus/DCASE2018/meta/test.csv"
with open(metaPath, "r") as metaFile:
    metadata = metaFile.read().splitlines()[1:]

metadata = [i.split("\t") for i in metadata]

# load the features
featTest = []
for file in featTestList:
    path = os.path.join(featTestPath, file)
    feature = np.load(path)

    # preprocessing
    feature = np.expand_dims(feature, axis=-1)
    featTest.append(feature)

featTest = np.array(featTest)


# # Load the models

# In[3]:


modelJsonPath = "/homeLocal/eriador/Documents/DCASE2018/results/testing/mel_old_noBlank_timeReduction2/oldMel_noBlank_reduce2_model.json"
modelWeightPath = "/homeLocal/eriador/Documents/DCASE2018/results/testing/mel_old_noBlank_timeReduction2/oldMel_noBlank_reduce2_weight.h5py"


# # Custom implementation for GRU and GRUCell

# In[4]:


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
                h_tm1_z = h_tm1 * self.temporal_weight #rec_dp_mask[0]
                h_tm1_r = h_tm1 * self.temporal_weight #rec_dp_mask[1]
                h_tm1_h = h_tm1 * self.temporal_weight #rec_dp_mask[2]
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




# # With GRU and TimeDistributed
# ## Remove the GlobalAveragePooling1D to trace score per frames

# In[5]:


##### load the model and perform prediction
def useClassic():
    K.clear_session()

    with open(modelJsonPath, "r") as modelJsonFile:
        model = model_from_json(modelJsonFile.read())
    model.load_weights(modelWeightPath)

    #model.summary()

    intermediate_model = Model(input=model.input, output=model.get_layer("time_distributed_1").output)
    #intermediate_model.summary()

    return model, intermediate_model

# retreive information about the custom
#model = useClassic()
#prediction = model.predict(featTest)
#print(prediction.shape)
#print(prediction[0].shape)
#print(prediction[0])


# # With ***Custom*** GRU and TimeDistributed
# ## Remove the GlobalAveragePooling1D to trace score per frames

# In[6]:


##### load the model and perform prediction
def useCustomGRU(temporalWeight: float) -> Model:
    K.clear_session()

    with open(modelJsonPath, "r") as modelJsonFile:
        print(1)
        model = model_from_json(modelJsonFile.read())
    print(1)
    model.load_weights(modelWeightPath)

    #disasemble layers
    print(1)
    layers = [l for l in model.layers]

    # Get the trained forward layer from the bidirectional and change it's property
    print(1)
    b1 = model.get_layer("bidirectional_1")

    x = layers[0].output
    print(1)
    for i in range(1, len(layers)):
        print(i, "/", len(layers), layers[i])
        if layers[i].name == "bidirectional_1":
            x = Bidirectional(
                CustomGRU(units=64, kernel_initializer='glorot_uniform', recurrent_dropout=0.8, dropout=0.0, return_sequences=True, temporal_weight=temporalWeight), name="custom_bi")(x)
        elif layers[i].name == "time_distributed_1":
            x = TimeDistributed( Dense(10, activation="sigmoid"), )(x)
        else:
            x = layers[i](x)

    print(1)
    newModel = Model(input=layers[0].input, output=x)
    print(2)
    newModel.load_weights(modelWeightPath)
    #model.summary()

    print(3)
    intermediate_model = Model(input=model.input, output=newModel.get_layer("time_distributed_1").output)
    #intermediate_model.summary()

    return intermediate_model

# retreive inform,ation about the custom
#model = useCustomGRU(0.2)
#prediction = model.predict(featTest)


# # Calculate the score using the **baseline tool**

# In[ ]:


def basic(method: str = "threshold", **kwargs):
    e = Encoder()
    gModel, tModel = useClassic()
    print("predicting ...")
    gPrediction = gModel.predict(featTest)
    tPrediction = tModel.predict(featTest)
    print(tPrediction.shape)
    print("encoding ...")
    kwargs["global_prediction"] = gPrediction
    segments = e.encode(tPrediction, method=method, **kwargs)
    print("evaluation ...")
    evaluation = e.parse(segments, featTestList)

    # write the evaluation on the disk
    with open("perso_eval.csv", "w") as f:
        f.write("filename\tonset\toffset\tevent_label\n")
        f.write(evaluation)

    perso_event_list = MetaDataContainer()
    perso_event_list.load(filename="perso_eval.csv")

    ref_event_list = MetaDataContainer()
    ref_event_list.load(filename="../../meta/test.csv")

    event_based_metric = event_based_evaluation(ref_event_list, perso_event_list)

    print(event_based_metric)

def printTable(results: list):

    print("%-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s |" % (
        22, "model", 22, "Global F-measure", 22, "Global Precision", 22, "Global Recall", 22, "ER",
        22, "Class-wise F-measure", 22, "Class-wise Precision", 22, "Class-wise Recall", 22, "ER"))
    print(("-" * 22 + " | ") * 9)

    for weight, info in results:
        print("%-*s | " % (22, "CGRU %.2f" % (weight)), end="")
        print("%-*s | " % (22, "%.2f %%" % (info["overall"]["f_measure"]["f_measure"] * 100)), end="")
        print("%-*s | " % (22, "%.2f %%" % (info["overall"]["f_measure"]["precision"] * 100)), end="")
        print("%-*s | " % (22, "%.2f %%" % (info["overall"]["f_measure"]["recall"] * 100)), end="")
        print("%-*s | " % (22, "%.2f " % (info["overall"]["error_rate"]["error_rate"])), end="")
        print("%-*s | " % (22, "%.2f %%" % (info["class-wise"]["f_measure"]["f_measure"] * 100)), end="")
        print("%-*s | " % (22, "%.2f %%" % (info["class-wise"]["f_measure"]["precision"] * 100)), end="")
        print("%-*s | " % (22, "%.2f %%" % (info["class-wise"]["f_measure"]["recall"] * 100)), end="")
        print("%-*s | " % (22, "%.2f " % (info["class-wise"]["error_rate"]["error_rate"])), end="")
        print("")

def evaluate(evaluation):
    # write the evaluation on the disk
    print("perform evaluation ...")
    with open("perso_eval.csv", "w") as f:
        f.write("filename\tonset\toffset\tevent_label\n")
        f.write(evaluation)

    perso_event_list = MetaDataContainer()
    perso_event_list.load(filename="perso_eval.csv")

    ref_event_list = MetaDataContainer()
    ref_event_list.load(filename="../../meta/test.csv")

    event_based_metric = event_based_evaluation(ref_event_list, perso_event_list)
    #print(event_based_metric)

    result = {
        "overall": event_based_metric.results_overall_metrics(),
        "class-wise": event_based_metric.results_class_wise_average_metrics(),
    }

    return result


def advanced(temporal_weights: list, method="threshold", smooth="smoothMovingAvg", **kwargs) -> list:
    results = []

    # classic ----------------------
    print("loading model ...")
    _, model = useClassic()
    print("predicting ...")
    prediction = model.predict(featTest)

    e = Encoder()
    print("encoding results ...")
    segments = e.encode(prediction, method=method, smooth=smooth, **kwargs)
    evaluation = e.parse(segments, featTestList)

    results.append((1.0, evaluate(evaluation)))

    # custom GRU prediction - ---------------------
    for weight in temporal_weights:
        print("loading model ...")
        model = useCustomGRU(weight)
        print("predicting ...")
        prediction = model.predict(featTest)

        e = Encoder()
        print("encoding results ...")
        segments = e.encode(prediction, method=method, smooth=smooth, **kwargs)
        evaluation = e.parse(segments, featTestList)

        results.append((weight, evaluate(evaluation)))

    return results

gModel, tModel = useClassic()
print("predicting ...")
gPrediction = gModel.predict(featTest)

#print("HYSTERESIS")
#printTable( advanced([0.1, 0.2, 0.23, 0.25, 0.3, 0.5], method="hysteresis", smooth="smoothMovingAvg", high=0.6, low=0.4) )

print("PRIMITIVE")
printTable( advanced([0.1, 0.2, 0.23, 0.25, 0.3, 0.5], method="primitive", smooth="smoothMovingAvg", window_size=8, stride=1, thresshold=2) )

#print("THRESHOLD")
#printTable( advanced([0.1, 0.2, 0.23, 0.25, 0.3, 0.5], method="threshold", smooth="smoothMovingAvg") )

basic(method="mean-threshold")
#basic(method="hysteresis")
#basic(method="hysteresis", high=0.7, low=0.3)
#basic(method="hysteresis", high=0.8, low=0.2)
#basic(method="derivative")
#basic(method="primitive", window_size=10, stride=1, threshold=3)
#basic()


# # Different segmentation methods results for "old mel without blank" and 20 ms frame length
#
# ***
#
# ## *Threshold based segmentation with "hole filling"*
#
#  - **CGRU x.xx** : Custom Gate Recurent Unit usage with temporal weight of x.xx
#  - **GRU** : The classic GRU implementation
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 3.79 %                 | 3.15 %                 | 4.75 %                 | 2.38                   | 4.96 %                 | 3.56 %                 | 11.99 %                | 3.25                   |
# CGRU 0.10              | 6.61 %                 | 4.88 %                 | 10.26 %                | 2.88                   | 5.55 %                 | 6.68 %                 | 7.25 %                 | 5.18                   |
# CGRU 0.20              | 9.60 %                 | 7.19 %                 | 14.46 %                | 2.70                   | 8.25 %                 | 9.82 %                 | 10.43 %                | 4.78                   |
# CGRU 0.23              | 10.41 %                | 7.88 %                 | 15.34 %                | 2.62                   | 9.40 %                 | 10.85 %                | 11.86 %                | 4.61                   |
# CGRU 0.25              | 10.59 %                | 8.06 %                 | 15.45 %                | 2.59                   | 9.88 %                 | 16.22 %                | 12.17 %                | 4.50                   |
# CGRU 0.30              | 9.45 %                 | 7.32 %                 | 13.36 %                | 2.54                   | 9.23 %                 | 10.04 %                | 12.03 %                | 4.30                   |
# CGRU 0.50              | 4.23 %                 | 3.39 %                 | 5.63 %                 | 2.52                   | 4.47 %                 | 4.07 %                 | 9.26 %                 | 3.68                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#
#
#
# ## *Hysteresis based segmentation*
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 4.52 %                 | 4.40 %                 | 4.64 %                 | 1.93                   | 6.40 %                 | 5.05 %                 | 11.75 %                | 2.48                   |
# CGRU 0.10              | 2.94 %                 | 2.26 %                 | 4.19 %                 | 2.76                   | 2.27 %                 | 2.02 %                 | 3.60 %                 | 4.55                   |
# CGRU 0.20              | 5.05 %                 | 4.04 %                 | 6.73 %                 | 2.52                   | 4.61 %                 | 4.98 %                 | 5.72 %                 | 4.17                   |
# CGRU 0.23              | 5.40 %                 | 4.37 %                 | 7.06 %                 | 2.46                   | 4.44 %                 | 5.00 %                 | 5.82 %                 | 4.04                   |
# CGRU 0.25              | 5.41 %                 | 4.42 %                 | 6.95 %                 | 2.43                   | 4.37 %                 | 5.13 %                 | 5.83 %                 | 3.93                   |
# CGRU 0.30              | 5.93 %                 | 5.00 %                 | 7.28 %                 | 2.30                   | 5.52 %                 | 7.95 %                 | 6.75 %                 | 3.67                   |
# CGRU 0.50              | 3.60 %                 | 3.23 %                 | 4.08 %                 | 2.15                   | 3.83 %                 | 3.51 %                 | 7.90 %                 | 3.06                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#
#
#
# ## *Primitive based segmentation*
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 5.11 %                 | 4.25 %                 | 6.40 %                 | 2.33                   | 6.29 %                 | 4.60 %                 | 15.28 %                | 3.22                   |
# CGRU 0.10              | 7.46 %                 | 5.48 %                 | 11.70 %                | 2.88                   | 7.13 %                 | 8.26 %                 | 9.00 %                 | 5.21                   |
# CGRU 0.20              | 9.29 %                 | 6.95 %                 | 14.02 %                | 2.72                   | 9.14 %                 | 13.17 %                | 12.08 %                | 4.70                   |
# CGRU 0.23              | 9.19 %                 | 6.91 %                 | 13.69 %                | 2.69                   | 9.12 %                 | 10.09 %                | 12.34 %                | 4.59                   |
# CGRU 0.25              | 8.59 %                 | 6.53 %                 | 12.58 %                | 2.65                   | 8.79 %                 | 9.25 %                 | 12.07 %                | 4.45                   |
# CGRU 0.30              | 6.25 %                 | 4.77 %                 | 9.05 %                 | 2.69                   | 6.76 %                 | 6.77 %                 | 9.82 %                 | 4.31                   |
# CGRU 0.50              | 3.63 %                 | 2.89 %                 | 4.86 %                 | 2.55                   | 4.18 %                 | 3.67 %                 | 9.91 %                 | 3.72                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#

# # Different segmentation methods results for "old mel without blank" and **40** ms frame length
#
# ***
#
# ## *Threshold based segmentation with "hole filling"*
#
#  - **CGRU x.xx** : Custom Gate Recurent Unit usage with temporal weight of x.xx
#  - **GRU** : The classic GRU implementation
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 4.31 %                 | 3.95 %                 | 4.75 %                 | 2.07                   | 6.68 %                 | 5.33 %                 | 11.40 %                | 2.55                   |
# CGRU 0.10              | 12.96 %                | 10.94 %                | 15.89 %                | 2.12                   | 11.47 %                | 12.47 %                | 10.94 %                | 3.73                   |
# CGRU 0.20              | 13.47 %                | 11.69 %                | 15.89 %                | 2.02                   | 11.35 %                | 11.81 %                | 12.29 %                | 3.37                   |
# CGRU 0.23              | 12.73 %                | 11.17 %                | 14.79 %                | 2.01                   | 11.37 %                | 11.47 %                | 12.50 %                | 3.22                   |
# CGRU 0.25              | 12.57 %                | 11.05 %                | 14.57 %                | 2.01                   | 11.52 %                | 11.52 %                | 12.94 %                | 3.17                   |
# CGRU 0.30              | 9.25 %                 | 8.07 %                 | 10.82 %                | 2.10                   | 8.24 %                 | 8.37 %                 | 10.58 %                | 3.18                   |
# CGRU 0.50              | 4.00 %                 | 3.52 %                 | 4.64 %                 | 2.19                   | 4.69 %                 | 4.34 %                 | 7.81 %                 | 2.92                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#
#
#
# ## *Hysteresis based segmentation*
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 5.43 %                 | 6.15 %                 | 4.86 %                 | 1.67                   | 8.90 %                 | 8.53 %                 | 11.18 %                | 1.90                   |
# CGRU 0.10              | 6.91 %                 | 7.10 %                 | 6.73 %                 | 1.81                   | 5.79 %                 | 9.08 %                 | 5.75 %                 | 2.60                   |
# CGRU 0.20              | 7.91 %                 | 8.51 %                 | 7.40 %                 | 1.71                   | 6.40 %                 | 7.08 %                 | 7.27 %                 | 2.38                   |
# CGRU 0.23              | 7.83 %                 | 8.46 %                 | 7.28 %                 | 1.71                   | 6.49 %                 | 7.02 %                 | 7.32 %                 | 2.35                   |
# CGRU 0.25              | 7.91 %                 | 8.50 %                 | 7.40 %                 | 1.71                   | 7.13 %                 | 7.47 %                 | 8.23 %                 | 2.30                   |
# CGRU 0.30              | 7.39 %                 | 8.20 %                 | 6.73 %                 | 1.68                   | 7.44 %                 | 8.28 %                 | 8.57 %                 | 2.20                   |
# CGRU 0.50              | 3.57 %                 | 3.88 %                 | 3.31 %                 | 1.76                   | 4.52 %                 | 3.93 %                 | 7.21 %                 | 2.20                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#
#
#
# ## *Primitive based segmentation*
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 5.50 %                 | 5.49 %                 | 5.52 %                 | 1.84                   | 8.53 %                 | 7.33 %                 | 13.40 %                | 2.27                   |
# CGRU 0.10              | 10.17 %                | 8.87 %                 | 11.92 %                | 2.08                   | 9.32 %                 | 10.37 %                | 11.83 %                | 3.65                   |
# CGRU 0.20              | 8.08 %                 | 7.16 %                 | 9.27 %                 | 2.09                   | 8.46 %                 | 8.37 %                 | 11.98 %                | 3.34                   |
# CGRU 0.23              | 7.83 %                 | 6.96 %                 | 8.94 %                 | 2.09                   | 8.27 %                 | 8.36 %                 | 11.90 %                | 3.25                   |
# CGRU 0.25              | 6.93 %                 | 6.15 %                 | 7.95 %                 | 2.12                   | 6.87 %                 | 6.92 %                 | 11.40 %                | 3.25                   |
# CGRU 0.30              | 6.60 %                 | 5.89 %                 | 7.51 %                 | 2.10                   | 6.76 %                 | 6.62 %                 | 11.17 %                | 3.09                   |
# CGRU 0.50              | 4.46 %                 | 4.04 %                 | 4.97 %                 | 2.09                   | 5.82 %                 | 4.85 %                 | 10.46 %                | 2.70                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#

# # Different segmentation methods results for "old mel without blank" and **80** ms frame length
#
# ***
#
# ## *Threshold based segmentation with "hole filling"*
#
#  - **CGRU x.xx** : Custom Gate Recurent Unit usage with temporal weight of x.xx
#  - **GRU** : The classic GRU implementation
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 4.33 %                 | 4.61 %                 | 4.08 %                 | 1.77                   | 6.53 %                 | 5.42 %                 | 9.41 %                 | 2.12                   |
# CGRU 0.10              | 11.70 %                | 13.06 %                | 10.60 %                | 1.59                   | 8.87 %                 | 10.75 %                | 6.77 %                 | 1.99                   |
# CGRU 0.20              | 9.80 %                 | 10.69 %                | 9.05 %                 | 1.65                   | 7.85 %                 | 9.15 %                 | 7.08 %                 | 2.07                   |
# CGRU 0.23              | 7.85 %                 | 8.67 %                 | 7.17 %                 | 1.67                   | 6.84 %                 | 8.49 %                 | 6.06 %                 | 2.09                   |
# CGRU 0.25              | 7.36 %                 | 8.12 %                 | 6.73 %                 | 1.68                   | 6.97 %                 | 8.34 %                 | 6.90 %                 | 2.09                   |
# CGRU 0.30              | 6.47 %                 | 6.78 %                 | 6.18 %                 | 1.76                   | 5.84 %                 | 6.97 %                 | 6.43 %                 | 2.24                   |
# CGRU 0.50              | 5.36 %                 | 5.32 %                 | 5.41 %                 | 1.87                   | 6.35 %                 | 6.11 %                 | 9.10 %                 | 2.39                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#
#
#
# ## *Hysteresis based segmentation*
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 4.86 %                 | 6.55 %                 | 3.86 %                 | 1.48                   | 8.28 %                 | 7.87 %                 | 9.51 %                 | 1.64                   |
# CGRU 0.10              | 6.41 %                 | 9.89 %                 | 4.75 %                 | 1.38                   | 4.84 %                 | 7.19 %                 | 3.29 %                 | 1.54                   |
# CGRU 0.20              | 5.82 %                 | 8.97 %                 | 4.30 %                 | 1.38                   | 4.74 %                 | 6.81 %                 | 3.86 %                 | 1.55                   |
# CGRU 0.23              | 4.96 %                 | 7.76 %                 | 3.64 %                 | 1.39                   | 4.71 %                 | 5.89 %                 | 4.38 %                 | 1.54                   |
# CGRU 0.25              | 4.48 %                 | 6.94 %                 | 3.31 %                 | 1.40                   | 4.25 %                 | 5.15 %                 | 3.89 %                 | 1.57                   |
# CGRU 0.30              | 4.73 %                 | 7.14 %                 | 3.53 %                 | 1.41                   | 5.41 %                 | 7.28 %                 | 5.09 %                 | 1.59                   |
# CGRU 0.50              | 4.55 %                 | 6.37 %                 | 3.53 %                 | 1.46                   | 6.14 %                 | 6.67 %                 | 7.23 %                 | 1.69                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#
#
#
# ## *Primitive based segmentation*
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 5.55 %                 | 6.48 %                 | 4.86 %                 | 1.60                   | 8.53 %                 | 7.37 %                 | 11.46 %                | 1.88                   |
# CGRU 0.10              | 3.81 %                 | 5.26 %                 | 2.98 %                 | 1.50                   | 3.43 %                 | 4.01 %                 | 3.97 %                 | 1.75                   |
# CGRU 0.20              | 3.86 %                 | 5.16 %                 | 3.09 %                 | 1.52                   | 4.45 %                 | 6.22 %                 | 5.04 %                 | 1.83                   |
# CGRU 0.23              | 3.77 %                 | 4.84 %                 | 3.09 %                 | 1.56                   | 4.64 %                 | 5.76 %                 | 5.55 %                 | 1.87                   |
# CGRU 0.25              | 4.48 %                 | 5.55 %                 | 3.75 %                 | 1.58                   | 5.00 %                 | 5.92 %                 | 6.96 %                 | 1.91                   |
# CGRU 0.30              | 3.33 %                 | 3.97 %                 | 2.87 %                 | 1.64                   | 3.94 %                 | 3.49 %                 | 6.56 %                 | 2.02                   |
# CGRU 0.50              | 4.24 %                 | 5.08 %                 | 3.64 %                 | 1.61                   | 6.15 %                 | 5.54 %                 | 8.60 %                 | 1.90                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#

# # Different segmentation methods results for "old mel without blank" and **160** ms frame length
#
# ***
#
# ## *Threshold based segmentation with "hole filling"*
#
#  - **CGRU x.xx** : Custom Gate Recurent Unit usage with temporal weight of x.xx
#  - **GRU** : The classic GRU implementation
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 5.85 %                 | 6.70 %                 | 5.19 %                 | 1.64                   | 8.61 %                 | 7.85 %                 | 11.91 %                | 1.94                   |
# CGRU 0.10              | 7.94 %                 | 9.45 %                 | 6.84 %                 | 1.57                   | 7.93 %                 | 11.06 %                | 6.29 %                 | 2.08                   |
# CGRU 0.20              | 6.91 %                 | 8.01 %                 | 6.07 %                 | 1.61                   | 6.22 %                 | 8.59 %                 | 6.18 %                 | 2.15                   |
# CGRU 0.23              | 6.67 %                 | 7.58 %                 | 5.96 %                 | 1.64                   | 6.19 %                 | 8.22 %                 | 6.56 %                 | 2.17                   |
# CGRU 0.25              | 7.33 %                 | 8.20 %                 | 6.62 %                 | 1.65                   | 6.92 %                 | 9.20 %                 | 7.16 %                 | 2.15                   |
# CGRU 0.30              | 5.80 %                 | 6.40 %                 | 5.30 %                 | 1.69                   | 5.65 %                 | 7.10 %                 | 6.62 %                 | 2.16                   |
# CGRU 0.50              | 5.05 %                 | 5.71 %                 | 4.53 %                 | 1.67                   | 6.47 %                 | 6.82 %                 | 9.15 %                 | 2.00                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#
#
#
# ## *Hysteresis based segmentation*
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 5.97 %                 | 8.40 %                 | 4.64 %                 | 1.43                   | 9.51 %                 | 9.95 %                 | 11.05 %                | 1.61                   |
# CGRU 0.10              | 4.36 %                 | 7.39 %                 | 3.09 %                 | 1.35                   | 4.07 %                 | 6.28 %                 | 3.45 %                 | 1.53                   |
# CGRU 0.20              | 3.91 %                 | 6.70 %                 | 2.76 %                 | 1.35                   | 3.68 %                 | 5.44 %                 | 3.63 %                 | 1.51                   |
# CGRU 0.23              | 3.56 %                 | 5.96 %                 | 2.54 %                 | 1.37                   | 3.56 %                 | 5.53 %                 | 3.79 %                 | 1.55                   |
# CGRU 0.25              | 3.69 %                 | 6.09 %                 | 2.65 %                 | 1.37                   | 3.69 %                 | 5.42 %                 | 4.18 %                 | 1.57                   |
# CGRU 0.30              | 4.27 %                 | 6.93 %                 | 3.09 %                 | 1.37                   | 4.90 %                 | 6.96 %                 | 5.27 %                 | 1.54                   |
# CGRU 0.50              | 5.11 %                 | 7.98 %                 | 3.75 %                 | 1.38                   | 7.72 %                 | 9.36 %                 | 8.53 %                 | 1.50                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#
#
#
# ## *Primitive based segmentation*
#
# model                  | Global F-measure       | Global Precision       | Global Recall          | ER                     | Class-wise F-measure   | Class-wise Precision   | Class-wise Recall      | ER                     |
# ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
# GRU                    | 7.05 %                 | 9.12 %                 | 5.74 %                 | 1.48                   | 11.40 %                | 11.42 %                | 13.84 %                | 1.65                   |
# CGRU 0.10              | 2.69 %                 | 4.76 %                 | 1.88 %                 | 1.35                   | 3.05 %                 | 5.48 %                 | 4.27 %                 | 1.59                   |
# CGRU 0.20              | 3.30 %                 | 5.16 %                 | 2.43 %                 | 1.41                   | 4.18 %                 | 5.28 %                 | 6.12 %                 | 1.65                   |
# CGRU 0.23              | 3.88 %                 | 5.98 %                 | 2.87 %                 | 1.41                   | 5.33 %                 | 8.05 %                 | 6.92 %                 | 1.65                   |
# CGRU 0.25              | 4.14 %                 | 6.26 %                 | 3.09 %                 | 1.42                   | 5.54 %                 | 7.61 %                 | 7.23 %                 | 1.67                   |
# CGRU 0.30              | 4.82 %                 | 7.13 %                 | 3.64 %                 | 1.43                   | 6.35 %                 | 8.15 %                 | 8.51 %                 | 1.68                   |
# CGRU 0.50              | 6.29 %                 | 8.59 %                 | 4.97 %                 | 1.46                   | 8.64 %                 | 9.08 %                 | 12.33 %                | 1.69                   |
# Baseline | 14.06 % | - | - | 1.54 | - | - | - | - |
#
