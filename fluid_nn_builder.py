
from string import Template

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os

from typing_extensions import Self
from typing import Any, List

import unicodedata
import re

# ref: https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename
def slugify(value, allow_unicode = False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


# ================================== LAYERS ==================================
class LayerBuilder:
    def buildFirst(self, inputShape: tuple):
        pass

    def build(self):
        pass

    def getSize() -> int:
        pass

    def describe() -> str:
        pass


class Conv2DLayerBuilder(LayerBuilder):
    _size: int = 0
    _kernel: tuple = (0, 0)
    _padding: str = "same"
    _activation: str = "relu"

    def size(self, size) -> Self:
        self._size = size
        return self

    def kernel(self, kernel) -> Self:
        self._kernel = kernel
        return self

    def activation(self, activation) -> Self:
        self._activation = activation
        return self

    def padding(self, padding) -> Self:
        self._padding = padding
        return self

    def getSize(self) -> int:
        return self._size

    def describe(self) -> str:
        return 'Conv2D(' + str(self._size) + ', ' + str(self._kernel) + ', padding=' + str(self._padding) + ')'

    def buildFirst(self, inputShape):
        return Conv2D(self._size, self._kernel, padding = self._padding, activation = self._activation, input_shape = inputShape)

    def build(self):
        return Conv2D(self._size, self._kernel, padding = self._padding, activation = self._activation)


class DenseLayerBuilder(LayerBuilder):
    _size: int = 0
    _activation: str = "relu"

    def size(self, size) -> Self:
        self._size = size
        return self

    def activation(self, activation) -> Self:
        self._activation = activation
        return self

    def getSize(self) -> int:
        return self._size

    def describe(self) -> str:
        return 'Dense(' + str(self._size) + ')'

    def buildFirst(self, inputShape):
        return Dense(self._size, activation = self._activation, input_shape = inputShape)

    def build(self):
        return Dense(self._size, activation = self._activation)


class MaxPooling2DLayerBuilder(LayerBuilder):
    _kernel: tuple = (0, 0)

    def kernel(self, kernel) -> Self:
        self._kernel = kernel
        return self

    def getSize() -> int:
        raise Exception("unsupported")

    def describe(self) -> str:
        return 'MaxPooling2D(' + str(self._kernel) + ')'

    def buildFirst(self, inputShape):
        return MaxPooling2D(self._kernel, input_shape = inputShape)

    def build(self):
        return MaxPooling2D(self._kernel)


class DropoutLayerBuilder(LayerBuilder):
    _rate: float = 0.25

    def rate(self, rate) -> Self:
        self._rate = rate
        return self

    def getSize() -> int:
        raise Exception("unsupported")

    def describe(self) -> str:
        return 'Dropout(' + str(self._rate) + ')'

    def buildFirst(self, inputShape):
        return Dropout(self._rate, input_shape = inputShape)

    def build(self):
        return Dropout(self._rate)


class FlattenLayerBuilder(LayerBuilder):

    def getSize() -> int:
        raise Exception("unsupported")

    def describe(self) -> str:
        return 'Flatten'

    def buildFirst(self, inputShape):
        raise Exception("unsupported")

    def build(self):
        return Flatten()

class BatchNormalizationLayerBuilder(LayerBuilder):

    def getSize() -> int:
        raise Exception("unsupported")

    def describe(self) -> str:
        return 'BatchNormalization'

    def buildFirst(self, inputShape):
        return BatchNormalization(input_shape = inputShape)

    def build(self):
        return BatchNormalization()

# =================================================================================
# ================================== DEFINITIONS ==================================

class CompilerDef:
    _loss: str = None
    _optimizer: str = "adam"
    _learningRate: float = None
    _metrics: list = []

    def loss(self, loss: str) -> Self:
        self._loss = loss
        return self

    def optimizer(self, optimizer: str) -> Self:
        self._optimizer = optimizer
        return self

    def learningRate(self, learningRate: float) -> Self:
        self._learningRate = learningRate
        return self

    def metrics(self, metrics: list) -> Self:
        self._metrics = metrics
        return self

    def describe(self) -> str:
        return 'loss ' + self._loss + ', optimizer ' + self._optimizer + ', metrics ' + str(self._metrics) 

class FitterDef:
    _batchSize: int = 64
    _epochs: int = 10
    _verbose: bool = False

    def batchSize(self, batchSize: int) -> Self:
        self._batchSize = batchSize
        return self

    def epochs(self, epochs: int) -> Self:
        self._epochs = epochs
        return self

    def verbose(self, verbose: bool) -> Self:
        self._verbose = verbose
        return self

    def describe(self) -> str:
        return 'batchSize ' + str(self._batchSize) + ', epochs ' + str(self._epochs) 

class ValidationDef:
    _valFraction: float = None
    _valSet: tuple = None

    def valFraction(self, valFraction: float) -> Self:
        self._valFraction = valFraction
        return self

    def valSet(self, x_test, y_test) -> Self:
        self._valSet = (x_test, y_test)
        return self

    def describe(self) -> str:
        return 'valFraction ' + str(self._valFraction) if self._valFraction else 'valSet size ' + str(len(self._valSet[1]))

class PlotterDef:
    _useArchNameInTitle: bool = False
    _useLayersInTitle: bool = True
    _useCompilerDefInTitle: bool = False
    _useFitterDefInTitle: bool = True
    _useValidationDefInTitle: bool = True
    
    _lossName: str = 'loss'
    _runName: str = 'run'
    _finalMetricsOnEvalSetName = 'final $metrics on evaluation set:'
    _finalLossOnEvalSetName = 'final loss on evaluation set:'
    _correlationName = 'correlation'
    _legendMetricsOnLearningSet = '$metrics on learning set'
    _legendMetricsOnValidationSet = '$metrics on validation set'
    _legendLossOnLearningSet = 'loss on learning set'
    _legendLossOnValidationSet = 'loss on validation set'

    _subFolder: str = 'figures'

    def withArchName(self, useArchNameInTitle: bool) -> Self:
        self._useArchNameInTitle = useArchNameInTitle
        return self

    def withLayers(self, useLayers: bool) -> Self:
        self._useLayersInTitle = useLayers
        return self

    def withCompilerDef(self, useCompilerDef: bool) -> Self:
        self._useCompilerDefInTitle = useCompilerDef
        return self

    def withFitterDef(self, useFitterDef: bool) -> Self:
        self._useFitterDefInTitle = useFitterDef
        return self

    def withValidationDef(self, useValidationDef: bool) -> Self:
        self._useValidationDefInTitle = useValidationDef
        return self

    def withSubFolder(self, subFolder: str) -> Self:
        self._subFolder = subFolder
        return self 

    def composeTitle(self, layers: List[LayerBuilder], compilerDef: CompilerDef, fitterDef: FitterDef, validationDef: ValidationDef, nnName: str = None) -> str:
        title = []
        if self._useArchNameInTitle:
            title.append(nnName)

        if self._useLayersInTitle:
            title.append('layers ' + str([l.describe() for l in layers]))

        if self._useCompilerDefInTitle:
            title.append(compilerDef.describe())

        if self._useFitterDefInTitle:
            title.append(fitterDef.describe())

        if self._useValidationDefInTitle:
            title.append(validationDef.describe())

        return ' '.join(title)

# ===============================================================================
# ================================== ACTUATORS ==================================

class TrainedModelResult:
    _model: Sequential = None
    _history = None

    def __init__(self, model: Sequential = None, history = None) -> None:
        self._model = model
        self._history = history

class EvaluatedModelResult(TrainedModelResult):
    _scores: dict = None

    def __init__(self, model: Sequential = None, history = None, scores: dict = None) -> None:
        super().__init__(model, history)
        self._scores = scores

class HistoryModelRunsPlotter:
    _title: str = None
    _compilerDef: CompilerDef = None
    _plotterDef: PlotterDef = None
    _modelResults: List[TrainedModelResult] = None

    def __init__(self, 
                    title: str,
                    compilerDef: CompilerDef,
                    plotterDef: PlotterDef,
                    modelResults: List[TrainedModelResult]):
        self._title = title
        self._compilerDef = compilerDef
        self._plotterDef = plotterDef
        self._subFolder = plotterDef._subFolder
        self._modelResults = modelResults

    def __plotOneRun(self, axPltAccuracy, axPltLoss, history, scores, i: int, nRuns: int) -> None:

        if len(self._compilerDef._metrics) > 1:
            raise Exception("only one metric is supported, several provided " + str(self._compilerDef._metrics))

        metricsName = self._compilerDef._metrics[0]

        df = pd.DataFrame(history.history, columns=[metricsName, "val_" + metricsName, "loss", "val_loss"])

        corrMatrix = df.corr()

        axPltAccuracy.set_title(
            metricsName + (", " + self._plotterDef._runName + " " + str(i + 1) if nRuns > 1 else "") 
            + (" (" + Template(self._plotterDef._finalMetricsOnEvalSetName).substitute(metrics = metricsName) + " " + str(round(scores["accuracy"] * 100, 4)) + " %)" if scores is not None else "")
            + ", " + self._plotterDef._correlationName + " " + str(round(corrMatrix[metricsName]["val_" + metricsName], 2))
        )

        axPltAccuracy.plot(history.history[metricsName], label = Template(self._plotterDef._legendMetricsOnLearningSet).substitute(metrics = metricsName))
        axPltAccuracy.plot(history.history["val_" + metricsName], label = Template(self._plotterDef._legendMetricsOnValidationSet).substitute(metrics = metricsName))
        axPltAccuracy.set_ylabel(metricsName)
        axPltAccuracy.legend()

        axPltLoss.set_title(
            self._plotterDef._lossName + (", " + self._plotterDef._runName + " " + str(i + 1) if nRuns > 1 else "")
            + (" (" + self._plotterDef._finalLossOnEvalSetName + " " + str(round(scores["loss"], 4)) + ")" if scores is not None else "")
            + ", " + self._plotterDef._correlationName + " " + str(round(corrMatrix["loss"]["val_loss"], 2))
        )
        axPltLoss.plot(history.history["loss"], label = self._plotterDef._legendLossOnLearningSet)
        axPltLoss.plot(history.history["val_loss"], label = self._plotterDef._legendLossOnValidationSet)
        axPltLoss.set_ylabel(self._plotterDef._lossName)
        axPltLoss.legend()

    def __createFigure(self) -> None:
        nRuns = len(self._modelResults)
        f, axes = plt.subplots(nRuns, 2, sharex = False, sharey = False, figsize = (30, 17))
        f.suptitle(self._title)

        j = 0
        for i in range(0, len(self._modelResults)):
            axPltAccuracy = axes[j, 0] if nRuns > 1 else axes[0]
            axPltLoss = axes[j, 1] if nRuns > 1 else axes[1]
            mr: TrainedModelResult = self._modelResults[i]
            history = mr._history
            scores = None
            if isinstance(mr, EvaluatedModelResult):
                scores = mr._scores
            j += 1

            self.__plotOneRun(axPltAccuracy, axPltLoss, history, scores, i, nRuns)

        plt.subplots_adjust(left = 0.03, bottom = 0.03, right = 0.983, top = 0.88, wspace = 0.086, hspace = 0.445)

    def __createFileNameFromTitle(self) -> str:
        if not os.path.exists(self._subFolder):
            os.makedirs(self._subFolder)
        return self._subFolder + '/' + slugify(self._title)

    def plot(self) -> None:
        self.__createFigure()
        plt.show()

    def save(self) -> tuple:
        self.__createFigure()
        fileName = self.__createFileNameFromTitle()
        plt.savefig(fileName)
        return fileName

    def saveAndPlot(self) -> tuple:
        fileName = self.save()
        plt.show()
        return fileName

class ModelRunsPlotter:
    _nnName: str = None
    _layers: List[LayerBuilder] = None
    _compilerDef: CompilerDef = None
    _fitterDef: FitterDef = None
    _validationDef: ValidationDef = None
    _plotterDef: PlotterDef = PlotterDef()
    _modelResults: List[TrainedModelResult] = None
    _title: str = None

    def __init__(self, 
                    nnName: str, 
                    layers: List[LayerBuilder], 
                    compilerDef: CompilerDef, 
                    fitterDef: FitterDef,
                    validationDef: ValidationDef,
                    plotterDef: PlotterDef,
                    modelResults: List[TrainedModelResult]):
        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef
        self._fitterDef = fitterDef
        self._validationDef = validationDef
        self._plotterDef = plotterDef
        self._modelResults = modelResults

    def configurePlotter(self) -> HistoryModelRunsPlotter:
        return HistoryModelRunsPlotter(self.ejectTitle(), self._compilerDef, self._plotterDef, self._modelResults)

    def ejectTitle(self) -> str:
        return self._plotterDef.composeTitle(self._layers, self._compilerDef, self._fitterDef, self._validationDef, self._nnName)

class EvaluatedModel:
    _nnName: str = None
    _layers: List[LayerBuilder] = None
    _compilerDef: CompilerDef = None
    _fitterDef: FitterDef = None
    _validationDef: ValidationDef = None
    _evaluatedModelResult: EvaluatedModelResult = EvaluatedModelResult()

    def __init__(self, 
                    nnName: str, 
                    layers: List[LayerBuilder], 
                    compilerDef: CompilerDef, 
                    fitterDef: FitterDef, 
                    validationDef: ValidationDef,
                    model: Sequential, 
                    history,
                    scores: dict):
        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef
        self._fitterDef = fitterDef
        self._validationDef = validationDef
        self._evaluatedModelResult._model = model
        self._evaluatedModelResult._history = history
        self._evaluatedModelResult._scores = scores

    def ejectResult(self) -> EvaluatedModelResult:
        return self._evaluatedModelResult

    def plotterDef(self, plotterDef: PlotterDef) -> HistoryModelRunsPlotter:
        return ModelRunsPlotter(self._nnName, self._layers, self._compilerDef, self._fitterDef, self._validationDef, \
            plotterDef, [self._evaluatedModelResult]).configurePlotter()

class ModelEvaluator:
    _nnName: str = None
    _layers: List[LayerBuilder] = None
    _compilerDef: CompilerDef = None
    _fitterDef: FitterDef = None
    _validationDef: ValidationDef = None
    _trainedModelResult: TrainedModelResult = None
    _x_test = None
    _y_test = None

    def __init__(self, 
                    nnName: str, 
                    layers: List[LayerBuilder], 
                    compilerDef: CompilerDef, 
                    fitterDef: FitterDef, 
                    validationDef: ValidationDef,
                    trainedModelResult: TrainedModelResult, 
                    x_test, y_test):
        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef
        self._fitterDef = fitterDef
        self._validationDef = validationDef
        self._trainedModelResult = trainedModelResult
        self._x_test = x_test
        self._y_test = y_test

    def evaluate(self, verbose: bool = False) -> EvaluatedModel:
        scores = self._trainedModelResult._model.evaluate(self._x_test, self._y_test, verbose = verbose)
        return EvaluatedModel(self._nnName, self._layers, self._compilerDef, self._fitterDef, self._validationDef, \
            self._trainedModelResult._model, self._trainedModelResult._history, {"loss": scores[0], "accuracy": scores[1]})

class TrainedModel:
    _nnName: str = None
    _layers: List[LayerBuilder] = None
    _compilerDef: CompilerDef = None
    _fitterDef: FitterDef = None
    _validationDef: ValidationDef = None
    _trainedModelResult: TrainedModelResult = TrainedModelResult()

    def __init__(self, 
                    nnName: str, 
                    layers: List[LayerBuilder], 
                    compilerDef: CompilerDef, 
                    fitterDef: FitterDef, 
                    validationDef: ValidationDef,
                    model: Sequential, 
                    history):
        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef
        self._fitterDef = fitterDef
        self._validationDef = validationDef
        self._trainedModelResult._model = model
        self._trainedModelResult._history = history

    def evalSet(self, x_test, y_test) -> ModelEvaluator:
        return ModelEvaluator(self._nnName, self._layers, self._compilerDef, self._fitterDef, self._validationDef, self._trainedModelResult, x_test, y_test)

    def ejectResult(self) -> TrainedModelResult:
        return self._trainedModelResult

    def plotterDef(self, plotterDef: PlotterDef) -> HistoryModelRunsPlotter:
        return ModelRunsPlotter(self._nnName, self._layers, self._compilerDef, self._fitterDef, self._validationDef, \
            plotterDef, [self._trainedModelResult]).configurePlotter()


class MultipleEvaluatedModels:
    _nnName: str = None
    _layers: List[LayerBuilder] = None
    _compilerDef: CompilerDef = None
    _fitterDef: FitterDef = None
    _validationDef: ValidationDef = None
    _evaluatedModelResults: List[EvaluatedModelResult] = None

    def __init__(self, 
                    nnName: str, 
                    layers: List[LayerBuilder], 
                    compilerDef: CompilerDef, 
                    fitterDef: FitterDef,
                    validationDef: ValidationDef,
                    evaluatedModelResults: List[EvaluatedModelResult]):
        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef
        self._fitterDef = fitterDef
        self._validationDef = validationDef
        self._evaluatedModelResults = evaluatedModelResults

    def ejectResults(self) -> List[EvaluatedModelResult]:
        return self._evaluatedModelResults

    def plotterDef(self, plotterDef: PlotterDef) -> HistoryModelRunsPlotter:
        return ModelRunsPlotter(self._nnName, self._layers, self._compilerDef, self._fitterDef, self._validationDef, \
            plotterDef, self._evaluatedModelResults).configurePlotter()

class MultipleModelsEvaluator:
    _nnName: str = None
    _layers: List[LayerBuilder] = None
    _compilerDef: CompilerDef = None
    _fitterDef: FitterDef = None
    _validationDef: ValidationDef = None
    _evalSet: tuple = None
    _trainedModelResults: List[TrainedModelResult] = None

    def __init__(self, 
                    nnName: str, 
                    layers: List[LayerBuilder], 
                    compilerDef: CompilerDef, 
                    fitterDef: FitterDef,
                    validationDef: ValidationDef,
                    trainedModelResults: List[TrainedModelResult],
                    evalSet: tuple):
        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef
        self._fitterDef = fitterDef
        self._validationDef = validationDef
        self._trainedModelResults = trainedModelResults
        self._evalSet = evalSet

    def evaluate(self, verbose: bool = False) -> MultipleEvaluatedModels:
        evaluatedModelResults: List[TrainedModelResult] = []
        for tm in self._trainedModelResults:
            scores = tm._model.evaluate(self._evalSet[0], self._evalSet[1], verbose)
            evaluatedModelResults.append(EvaluatedModelResult(tm._model, tm._history, {"loss": scores[0], "accuracy": scores[1]}))
        return MultipleEvaluatedModels(self._nnName, self._layers, self._compilerDef, self._fitterDef, self._validationDef, evaluatedModelResults)



class MultipleTrainedModels:
    _nnName: str = None
    _layers: List[LayerBuilder] = None
    _compilerDef: CompilerDef = None
    _fitterDef: FitterDef = None
    _validationDef: ValidationDef = None
    _trainedModelResults: List[TrainedModelResult] = None

    def __init__(self, 
                    nnName: str, 
                    layers: List[LayerBuilder], 
                    compilerDef: CompilerDef, 
                    fitterDef: FitterDef,
                    validationDef: ValidationDef, 
                    trainedModelResults: List[TrainedModelResult]):
        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef
        self._fitterDef = fitterDef
        self._validationDef = validationDef
        self._trainedModelResults = trainedModelResults

    def evalSet(self, x_test, y_test) -> MultipleModelsEvaluator:
        return MultipleModelsEvaluator(self._nnName, self._layers, self._compilerDef, self._fitterDef, self._validationDef, self._trainedModelResults, (x_test, y_test))

    def ejectResults(self) -> List[TrainedModelResult]:
        return self._trainedModelResults

    def plotterDef(self, plotterDef: PlotterDef) -> HistoryModelRunsPlotter:
        return ModelRunsPlotter(self._nnName, self._layers, self._compilerDef, self._fitterDef, self._validationDef, \
            plotterDef, self._trainedModelResults).configurePlotter()

class ModelTrainer:
    _nnName: str = None
    _layers: List[LayerBuilder] = []
    _compilerDef: CompilerDef = None
    _fitterDef: FitterDef = None
    _x_train: list
    _y_train: list
    _validationDef: ValidationDef = ValidationDef()

    def __init__(self, 
                    nnName: str, 
                    layers: List[LayerBuilder], 
                    compilerDef: CompilerDef, 
                    fitterDef: FitterDef, 
                    x_train: list, 
                    y_train: list):

        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef
        self._fitterDef = fitterDef
        self._x_train = x_train
        self._y_train = y_train

    def valSet(self, x_val, y_val) -> Self:
        self._validationDef.valSet(x_val, y_val)
        return self

    def valFraction(self, valFraction) -> Self:
        self._validationDef.valFraction(valFraction)
        return self

    def validationDef(self, validationDef: ValidationDef) -> Self:
        self._validationDef = validationDef
        return self

    def __calculateValSet(self) -> None:
        if self._validationDef._valSet:
            return self._validationDef._valSet
        elif self._validationDef._valFraction:
            self._x_train, x_val, self._y_train, y_val = train_test_split(self._x_train, self._y_train, test_size = self._validationDef._valFraction, shuffle = True)
            return (x_val, y_val)
        else:
            return None

    def __doFitModel(self, inputShape: tuple, valSet: tuple):

        model: Sequential = ModelCompiler(self._nnName, self._layers, self._compilerDef).compile(inputShape)

        if valSet:
            history = model.fit(
                self._x_train,
                self._y_train,
                batch_size = self._fitterDef._batchSize,
                epochs = self._fitterDef._epochs,
                validation_data = valSet,
                verbose = self._fitterDef._verbose
            )
        else:
            history = model.fit(
                self._x_train,
                self._y_train,
                batch_size = self._fitterDef._batchSize,
                epochs = self._fitterDef._epochs,
                verbose = self._fitterDef._verbose
            )

        return (model, history)

    def fitModelAndRepeat(self, n: int) -> MultipleTrainedModels:
        
        valSet = self.__calculateValSet()
        inputShape = self._x_train[0].shape
        trainedModelResults = []
        for i in range(0, n):
            (model, history) = self.__doFitModel(inputShape, valSet)
            trainedModelResults.append(TrainedModelResult(model, history))
        
        return MultipleTrainedModels(self._nnName, self._layers, self._compilerDef, self._fitterDef, self._validationDef, trainedModelResults)

    def fitModel(self) -> TrainedModel:

        inputShape = self._x_train[0].shape
        valSet = self.__calculateValSet()

        (model, history) = self.__doFitModel(inputShape, valSet)

        return TrainedModel(self._nnName, self._layers, self._compilerDef, self._fitterDef, self._validationDef, model, history)


class ModelFitter:
    _nnName: str = None
    _layers: List[LayerBuilder]
    _compilerDef: CompilerDef
    _fitterDef: FitterDef

    def __init__(self, nnName: str, layers: List[LayerBuilder], compilerDef: CompilerDef, fitterDef: FitterDef):
        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef
        self._fitterDef = fitterDef

    def trainSet(self, x_train, y_train) -> ModelTrainer:
        return ModelTrainer(self._nnName, self._layers, self._compilerDef, self._fitterDef, x_train, y_train)

class FitterDefBuilder:
    _nnName: str = None
    _layers: List[LayerBuilder]
    _compilerDef: CompilerDef
    _fitterDef: FitterDef = FitterDef()

    def __init__(self, nnName: str, layers: List[LayerBuilder], compilerDef: CompilerDef) -> None:
        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef

    def batchSize(self, batchSize: int) -> Self:
        self._fitterDef.batchSize(batchSize)
        return self

    def epochs(self, epochs: int) -> Self:
        self._fitterDef.epochs(epochs)
        return self

    def verbose(self, verbose: bool) -> Self:
        self._fitterDef.verbose(verbose)
        return self

    def configureFitter(self) -> ModelFitter:
        return ModelFitter(self._nnName, self._layers, self._compilerDef, self._fitterDef)

class ModelCompiler:
    _nnName: str = None
    _layers: List[LayerBuilder]
    _compilerDef: CompilerDef = CompilerDef()

    def __init__(self, nnName: str, layers: List[LayerBuilder], compilerDef: CompilerDef):
        self._nnName = nnName
        self._layers = layers
        self._compilerDef = compilerDef

    def __createOptimizer(self) -> Any:
        if (self._compilerDef._learningRate):
            if (self._compilerDef._optimizer == 'adam'):
                return Adam(self._compilerDef._learningRate)
            else:
                raise Exception('learning rate specified for unknown optimizer: ' + self._compilerDef._optimizer + ', learning rate: ' + self._compilerDef._learningRate)
        
        return self._compilerDef._optimizer

    def compile(self, inputShape: tuple) -> Sequential:
        model = ConfiguredModel(self._layers, inputShape).build()
        if self._compilerDef.metrics is None or len(self._compilerDef._metrics) == 0:
            raise Exception('metrics for CompilerDef is obligatory')

        model.compile(loss = self._compilerDef._loss, optimizer = self.__createOptimizer(), metrics = self._compilerDef._metrics)
        return model

    def fitterDef(self, fitterDef: FitterDef) -> ModelFitter:
        return ModelFitter(self._nnName, self._layers, self._compilerDef, fitterDef)

    def fitter(self) -> FitterDefBuilder:
        return FitterDefBuilder(self._nnName, self._layers, self._compilerDef)


class ConfiguredModel:
    _layers: List[LayerBuilder] = []
    _inputShape: None

    def __init__(self, layers: List[LayerBuilder], inputShape: tuple):
        self._layers = layers
        self._inputShape = inputShape

    def build(self) -> Sequential:
        model = Sequential()
        model.add(self._layers[0].buildFirst(self._inputShape))
        for i in range(1, len(self._layers)):
            model.add(self._layers[i].build())

        return model


# ==================================================================================
# ================================== ARCHITECTURE ==================================

class CompilerDefBuilder:
    _name: str = None
    _layers: List[LayerBuilder] = []
    _compilerDef: CompilerDef = CompilerDef()

    def __init__(self, name: str, layers: List[LayerBuilder]) -> None:
        self._name = name
        self._layers = layers

    def loss(self, loss: str) -> Self:
        self._compilerDef.loss(loss)
        return self

    def optimizer(self, optimizer: str) -> Self:
        self._compilerDef.optimizer(optimizer)
        return self

    def learningRate(self, learningRate: float) -> Self:
        self._compilerDef.learningRate(learningRate)
        return self

    def metrics(self, metrics: list) -> Self:
        self._compilerDef.metrics(metrics)
        return self

    def configureCompiler(self) -> ModelCompiler:
        return ModelCompiler(self._name, self._layers, self._compilerDef)


class NnArch:
    _name: str = None
    _layers: List[LayerBuilder] = []

    def __init__(self, layers, name = None):
        self._name = name
        self._layers = layers

    def add(self, lb: LayerBuilder) -> Self:
        self._layers.append(lb)
        return self

    def composeTitle(self, compilerDef: CompilerDef, fitterDef: FitterDef, validationDef: ValidationDef) -> str:
        return PlotterDef().composeTitle(self._layers, compilerDef, fitterDef, validationDef)

    def build(self, inputShape: tuple) -> ConfiguredModel:
        return ConfiguredModel(self._layers, inputShape).build()

    def compilerDef(self, compilerDef: CompilerDef) -> ModelCompiler:
        return ModelCompiler(self._name, self._layers, compilerDef)

    def compiler(self) -> CompilerDefBuilder:
        return CompilerDefBuilder(self._name, self._layers)


# ==================================================================================
# ================================== ARCHITECTURE ==================================


class LayerDefBuilder:
    @staticmethod
    def Conv2D():
        return Conv2DLayerBuilder()

    @staticmethod
    def MaxPooling2D():
        return MaxPooling2DLayerBuilder()

    @staticmethod
    def Dropout():
        return DropoutLayerBuilder()

    @staticmethod
    def Flatten():
        return FlattenLayerBuilder()

    @staticmethod
    def Dense():
        return DenseLayerBuilder()

    @staticmethod
    def BatchNormalization():
        return BatchNormalizationLayerBuilder()

def layers():
    return LayerDefBuilder()

def createCompilerDef():
    return CompilerDef()

def createFitterDef() -> FitterDef:
    return FitterDef()

def createValidationDef() -> ValidationDef:
    return ValidationDef()

def createPlotterDef() -> PlotterDef:
    return PlotterDef()

