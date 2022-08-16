from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras import utils
from typing import List

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import fluid_nn_builder as fnb

import unittest

(x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
y_train_full = utils.to_categorical(y_train_full, 10)
y_test_full = utils.to_categorical(y_test_full, 10)
x_train_full = x_train_full.reshape(x_train_full.shape[0], 28, 28, 1)
x_test_full = x_test_full.reshape(x_test_full.shape[0], 28, 28, 1)

print(x_train_full.shape)
print(x_test_full.shape)
print(y_train_full.shape)
print(y_test_full.shape)

trainSetFraction = 2000
x_train = x_train_full[0:trainSetFraction, :, :, :]
y_train = y_train_full[0:trainSetFraction, :]
x_test  = x_test_full[0:trainSetFraction, :, :, :]
y_test  = y_test_full[0:trainSetFraction, :]

architecture1 = fnb.NnArch([
        fnb.layers().Conv2D().size(32).kernel((3, 3)).padding('same').activation('relu'),
        fnb.layers().Conv2D().size(32).kernel((3, 3)).padding('same').activation('relu'),
        fnb.layers().MaxPooling2D().kernel((2, 2)),
        fnb.layers().Dropout().rate(0.25),
        fnb.layers().Flatten(),
        fnb.layers().Dense().size(256).activation('relu'),
        fnb.layers().Dropout().rate(0.25),
        fnb.layers().Dense().size(10).activation('softmax'),
    ], 'architecture 1'
)

batchSize = 512
nEpochs = 5


refModel = Sequential()
refModel.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape=(28, 28, 1)))
refModel.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
refModel.add(MaxPooling2D(pool_size=(2, 2)))
refModel.add(Dropout(0.25))
refModel.add(Flatten())
refModel.add(Dense(256, activation = 'relu'))
refModel.add(Dropout(0.25))
refModel.add(Dense(10, activation = 'softmax'))

refModel.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

refModel.fit(x_train, y_train, batch_size = batchSize, epochs = nEpochs, verbose = 1)


class TestModelEvaluating(unittest.TestCase):

    def test_modelFunctioning_evaluatingSingleRun(self):
        # given
        categoricalCompiler = fnb.createCompilerDef().loss('categorical_crossentropy').optimizer('adam').metrics(['accuracy'])
        fitterDef = fnb.createFitterDef().batchSize(batchSize).epochs(nEpochs).verbose(True)

        # when
        evaluatedModelResult: fnb.EvaluatedModelResult = architecture1.compilerDef(categoricalCompiler).fitterDef(fitterDef).trainSet(x_train, y_train)\
            .valFraction(0.2).fitModel().evalSet(x_test, y_test).evaluate().ejectResult()

        # then
        scores = evaluatedModelResult._scores
        refScores = refModel.evaluate(x_test, y_test)

        self.assertGreater(scores['loss'], 0.5)
        self.assertGreater(scores['accuracy'], 0.5)

        self.assertLess(abs(scores['loss'] - refScores[0]), 0.5)
        self.assertLess(abs(scores['accuracy'] - refScores[1]), 0.5)

    def test_modelFunctioning_evaluatingMultipleRuns(self):
        # given
        categoricalCompiler = fnb.createCompilerDef().loss('categorical_crossentropy').optimizer('adam').metrics(['accuracy'])
        fitterDef = fnb.createFitterDef().batchSize(batchSize).epochs(nEpochs).verbose(True)

        # when
        evaluatedModelResults: List[fnb.EvaluatedModelResult] = architecture1.compilerDef(categoricalCompiler).fitterDef(fitterDef).trainSet(x_train, y_train)\
            .valFraction(0.2).fitModelAndRepeat(2).evalSet(x_test, y_test).evaluate().ejectResults()

        # then
        self.assertEqual(len(evaluatedModelResults), 2)
        refScores = refModel.evaluate(x_test, y_test)
        for em in evaluatedModelResults:
            scores = em._scores

            self.assertGreater(scores['loss'], 0.5)
            self.assertGreater(scores['accuracy'], 0.5)

            self.assertLess(abs(scores['loss'] - refScores[0]), 0.5)
            self.assertLess(abs(scores['accuracy'] - refScores[1]), 0.5)




if __name__ == '__main__':
    unittest.main()