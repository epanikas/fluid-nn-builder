from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras import utils

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import fluid_nn_builder as fnb

import unittest


(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

architecture1 = fnb.NnArch([
        fnb.layers().Conv2D().size(32).kernel((3, 3)).padding('same').activation('relu'),
        fnb.layers().Conv2D().size(32).kernel((3, 3)).padding('same').activation('relu'),
        fnb.layers().MaxPooling2D().kernel((3, 3)),
        fnb.layers().Dropout().rate(0.25),
        fnb.layers().Flatten(),
        fnb.layers().Dense().size(256).activation('relu'),
        fnb.layers().Dropout().rate(0.25),
        fnb.layers().Dense().size(10).activation('softmax'),
    ], 'architecture 1'
)

batchSize = 512

categoricalCompiler = fnb.createCompilerDef().loss('categorical_crossentropy').optimizer('adam').metrics(['accuracy'])

oneEpochFitter = fnb.createFitterDef().batchSize(batchSize).epochs(1).verbose(True)

plotterDef = fnb.createPlotterDef()

class TestArchitecture(unittest.TestCase):

    def test_standardModelBuild(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation = 'softmax'))

        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        model.summary()

        history = model.fit(x_train, y_train, batch_size = batchSize, epochs = 1, verbose = 1)

        scores = model.evaluate(x_test, y_test)

        self.assertEqual(len(history.history['loss']), 1)
        self.assertEqual(len(scores), 2)
        

    def test_architectureBuild(self):
        model: Sequential = architecture1.build((28, 28, 1))

        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        history = model.fit(x_train, y_train, batch_size = batchSize, epochs = 1, verbose = 1)

        self.assertEqual(len(history.history['loss']), 1)

    def test_architectureBuildAndCompile(self):
        model: Sequential = architecture1.compilerDef(categoricalCompiler).compile((28, 28, 1))
        history = model.fit(x_train, y_train, batch_size = batchSize, epochs = 1, verbose = 1)

        self.assertEqual(len(history.history['loss']), 1)

    def test_architectureBuildCompileAndFitWithout_valSet(self):
        res = architecture1.compilerDef(categoricalCompiler).fitterDef(oneEpochFitter).trainSet(x_train, y_train).fitModel().ejectResult()
        model, history = res._model, res._history

        self.assertEqual(len(history.history['loss']), 1)

    def test_architectureBuildCompileAndFit_with_valSet(self):
        res = architecture1.compilerDef(categoricalCompiler).fitterDef(oneEpochFitter).trainSet(x_train, y_train).valSet(x_test, y_test).fitModel().ejectResult()
        model, history = res._model, res._history

        self.assertEqual(len(history.history['loss']), 1)

if __name__ == '__main__':
    unittest.main()