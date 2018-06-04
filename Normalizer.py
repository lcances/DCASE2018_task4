import numpy as np


def File_MinMaxNormalization(data: np.array) -> np.array:
    """
    Perform a Min Max normalization on each file independently.
    :param data: N-dimension array to normalize
    :return data: N-dimension array locally normalized
    """
    result = []

    for d in data:
        maxi = d.max(axis=0)
        mini = d.min(axis=0)

        result.append((d - mini) / (maxi - mini))

    return np.array(result)

def Global_MinMaxNormalization(data: np.array) -> np.array:
    """
    Perform a Min Max normalization of the whole dataset. The max and min are computed over the complete set of data
    :param data: N-dimension array to normalize
    :return data: N-dimension array globally normalized
    """
    maxi = data.max(axis=0)
    mini = data.min(axis=0)

    data = (data - mini) / (maxi - mini)

    return data

def File_MeanNormalization(data: np.array) -> np.array:
    """
    Perform a Mean normalization of each file independently.
    :param data: N-dimension array to normalize
    :return: data: N-dimension array locally normalized
    """

    result = []

    for d in data:
        mean = d.mean(axis=0)

        result.append(d - mean)

    return np.array(result)

def Global_MeanNormalization(data: np.array) -> np.array:
    """
    Perfor a Mean normalization of the whole dataset. The max and min are computed over the complete set of data
    :param data: N-dimension array to normalize
    :return: N-dimension array glovally normalized
    """
    mean = data.mean(axis=0)

    data = data - mean

    return data

def File_Standardization(data: np.array) -> np.array:
    """
    Perform a file wise standardiztion
    :param data: N-dimension array to normalize
    :return: data: N-dimension array locally normalized
    """

    result = []

    for d in data:
        mean = d.mean(axis=0)
        var = d.var(axis=0)

        result.append((d - mean) / var)

    return np.array(result)


def Global_Stadardization(data: np.array) -> np.array:
    """
    Perform a standardization on the whole dataset
    :param data: N-dimension array to normalize
    :return: N-dimension array globally normalized
    """
    mean = data.mean(axis=0)
    var = data.var(axis=0)

    data = (data - mean) / var

    return data


def UnitLength(data: np.array, order: int =None) -> np.array:
    return data / np.linalg.norm(data, ord=order, axis=1)


if __name__=='__main__':
    import random

    dataset = []
    for i in range(10):
        dataset.append([random.randint(0, 10) for j in range(10)])

    print("dataset =====")
    print("mean: ", np.mean(dataset))
    print("var: ", np.var(dataset))
    print("max: ", np.max(dataset))
    print("min: ", np.min(dataset))

    print("File MinMax")
    print(File_MinMaxNormalization(dataset))
    print("")

    print("File Mean")
    print(File_MeanNormalization(dataset))
    print("")

    print("File_Standardization")
    print(File_Standardization(dataset))
    print("")
