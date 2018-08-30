import numpy as np
from librosa import power_to_db


class WrongShapeException(Exception):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)


class NotFitYet(Exception):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)


class MethodsDoesNotExist(Exception):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)


class Scaler:
    def __init__(self, methods: str = "local"):
        self.methods = methods

    def fit(self, m: np.array):
        raise NotImplementedError

    def transform(self, m: np.array) -> np.array:
        raise NotImplementedError

    def __2d_transform(self, m: np.array) -> np.array:
        raise NotImplementedError

    def __3d_transform(self, m: np.array) -> np.array:
        raise NotImplementedError

    def fit_transform(self, m: np.array) -> np.array:
        raise NotImplementedError


class MinMaxScaler(Scaler):
    def __init__(self, methods: str = "local"):
        super().__init__(methods)
        self.mini = []
        self.maxi = []

    def fit(self, m: np.array, force: bool = False):
        if (self.maxi == [] and self.mini == []) or force:
            if self.methods == "global":
                self.mini = m.min(axis=1).min(axis=0)
                self.maxi = m.max(axis=1).max(axis=0)

            elif self.methods == "local":
                self.mini = m.min(axis=0)
                self.maxi = m.max(axis=0)

            else:
                raise MethodsDoesNotExist("Only available methods: \"global\" or \"local\"")

    def transform(self, m: np.array) -> np.array:
        if self.methods == "global":
            return self.__3d_transform(m)
        elif self.methods == "local":
            return self.__2d_transform(m)
        else:
            raise MethodsDoesNotExist("Only available methods: \"global\" or \"local\"")

    def fit_transform(self, m: np.array) -> np.array:
        self.fit(m)
        return self.transform(m)

    def __3d_transform(self, m: np.array):
        if self.mini == [] or self.maxi == []:
            raise NotFitYet("Data haven't been fit yet, can't perform scaling")

        if len(m.shape) != 3:
            raise WrongShapeException("Matrix should be of dimension 3")

        results = []

        for d in m:
            results.append((d - self.mini) / (self.maxi - self.mini))

        return np.array(results)

    def __2d_transform(self, m: np.array):
        results = []

        for d in m:
            results.append((d - d.min(axis=0)) / (d.max(axis=0) - d.min(axis=0)))

        return np.array(results)


class MeanScaler(Scaler):
    def __init__(self, methods: str = "local"):
        super().__init__(methods)
        self.mean = []

    def fit(self, m: np.array, force: bool = False):
        if self.mean == [] or force:
            if self.methods == "global":
                self.mean = m.mean(axis=1).mean(axis=0)

            elif self.methods == "local":
                self.mean = m.mean(axis=0)

            else:
                raise MethodsDoesNotExist("Only available methods: \"global\" or \"local\"")

    def transform(self, m: np.array) -> np.array:
        if self.methods == "global":
            return self.__3d_transform(m)

        elif self.methods == "local":
            return self.__2d_transform(m)

        else:
            raise MethodsDoesNotExist("Only available methods: \"global\" or \"local\"")

    def fit_transform(self, m: np.array) -> np.array:
        self.fit(m)
        return self.transform(m)

    def __3d_transform(self, m: np.array):
        if not self.mean:
            raise NotFitYet("Data haven't been fit yet, can't perform scaling")

        if len(m.shape) != 3:
            raise WrongShapeException("Matrix should be of dimension 3")

        results = []

        for d in m:
            results.append(d - self.mean)

        return np.array(results)

    def __2d_transform(self, m: np.array):
        results = []

        for d in m:
            results.append(d - d.mean(axis=0))

        return np.array(results)


def file_mean_normalization(data: np.array) -> np.array:
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


def global_mean_normalization(data: np.array) -> np.array:
    """
    Perfor a Mean normalization of the whole dataset. The max and min are computed over the complete set of data
    :param data: N-dimension array to normalize
    :return: N-dimension array glovally normalized
    """
    mean = data.mean(axis=0)

    data = data - mean

    return data


class StandardScaler(Scaler):
    def __init__(self, methods: str = "local"):
        super().__init__(methods)
        self.std = []
        self.mean = []

    def fit(self, m: np.array, force: bool = False):
        if (self.mean == [] and self.std == []) or force:
            if self.methods == "global":
                self.mean = m.mean(axis=1).mean(axis=0)
                self.std = m.std(axis=1).std(axis=0)

            elif self.methods == "local":
                self.mean = m.mean(axis=0)
                self.std = m.std(axis=0)

            else:
                raise MethodsDoesNotExist("Only available methods: \"global\" or \"local\"")

    def transform(self, m: np.array) -> np.array:
        if self.methods == "global":
            return self.__3d_transform(m)

        elif self.methods == "local":
            return self.__2d_transform(m)

        else:
            raise MethodsDoesNotExist("Only available methods: \"global\" or \"local\"")

    def fit_transform(self, m: np.array) -> np.array:
        self.fit(m)
        return self.transform(m)

    def __3d_transform(self, m: np.array):
        if not self.mean:
            raise NotFitYet("Data haven't been fit yet, can't perform scaling")

        if len(m.shape) != 3:
            raise WrongShapeException("Matrix should be of dimension 3")

        results = []

        for d in m:
            results.append((d - self.mean) / self.std)

        return np.array(results)

    def __2d_transform(self, m: np.array):
        results = []

        for d in m:
            results.append((d - d.mean(axis=0)) / d.std(axis=0))

        return np.array(results)


def file_standardization(data: np.array) -> np.array:
    """
    Perform a file wise standardization
    :param data: N-dimension array to normalize
    :return: data: N-dimension array locally normalized
    """

    result = []

    for d in data:
        mean = d.mean(axis=0)
        var = d.var(axis=0)

        result.append((d - mean) / var)

    return np.array(result)


def global_standardization(data: np.array) -> np.array:
    """
    Perform a standardization on the whole dataset
    :param data: N-dimension array to normalize
    :return: N-dimension array globally normalized
    """
    mean = data.mean(axis=0)
    var = data.var(axis=0)

    data = (data - mean) / var

    return data


class UnitScaler(Scaler):
    def __init__(self, methods: str = "local"):
        super().__init__(methods)

    def fit(self, m: np.array, force: bool = False):
        pass

    def transform(self, m: np.array) -> np.array:
        results = []

        for d in m:
            results.append(d / np.linalg.norm(d, ord=None, axis=0))

        return np.array(results)

    def fit_transform(self, m: np.array) -> np.array:
        self.fit(m)
        return self.transform(m)


def unit_length(data: np.array, order: int = None) -> np.array:
    result = []

    for d in data:
        result.append(d / np.linalg.norm(d, ord=order, axis=0))

    return np.array(result)


class LogScaler(Scaler):
    def __init__(self, methods: str = "local"):
        super().__init__(methods)

    def fit(self, m: np.array, force: bool = False):
        pass

    def transform(self, m: np.array) -> np.array:
        results = []

        for d in m:
            results.append(power_to_db(d))

        return np.array(results)

    def fit_transform(self, m: np.array) -> np.array:
        self.fit(m)
        return self.transform(m)
