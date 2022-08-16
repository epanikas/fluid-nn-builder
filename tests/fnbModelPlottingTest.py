from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras import utils

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
        fnb.layers().MaxPooling2D().kernel((2, 2)),
        fnb.layers().Flatten(),
        fnb.layers().Dense().size(256).activation('relu'),
        fnb.layers().Dropout().rate(0.25),
        fnb.layers().Dense().size(10).activation('softmax'),
    ], 'architecture 1'
)

batchSize = 512
nEpochs = 5

categoricalCompiler = fnb.createCompilerDef().loss('categorical_crossentropy').optimizer('adam').metrics(['accuracy'])
fitterDef = fnb.createFitterDef().batchSize(batchSize).epochs(nEpochs).verbose(True)
plotterDef = fnb.PlotterDef().withArchName(True).withLayers(True).withCompilerDef(False).withFitterDef(True).withValidationDef(True)\
    .withSubFolder('test-figures')

class TestModelPlotting(unittest.TestCase):

    def test_modelFunctioning_plottingSingleRun_withoutEvaluation(self):

        # when
        plotter: fnb.ModelRunsPlotter = architecture1.compilerDef(categoricalCompiler).fitterDef(fitterDef).trainSet(x_train, y_train)\
            .valFraction(0.2).fitModel().plotterDef(plotterDef)

        # then        
        title = plotter._title
        fileName = plotter.save()

        self.assertEqual(title, "architecture 1 layers ['Conv2D(32, (3, 3), padding=same)', 'MaxPooling2D((2, 2))', 'Flatten', 'Dense(256)', 'Dropout(0.25)', 'Dense(10)'] batchSize 512, epochs 5 valFraction 0.2")
        self.assertEqual(fileName, "test-figures/architecture-1-layers-conv2d32-3-3-paddingsame-maxpooling2d2-2-flatten-dense256-dropout025-dense10-batchsize-512-epochs-5-valfraction-02")

    def test_modelFunctioning_plottingSingleRun_withEvaluation(self):

        # when
        plotter: fnb.ModelRunsPlotter = architecture1.compilerDef(categoricalCompiler).fitterDef(fitterDef).trainSet(x_train, y_train)\
            .valFraction(0.2).fitModel().evalSet(x_test, y_test).evaluate().plotterDef(plotterDef)

        # then        
        title = plotter._title
        fileName = plotter.save()

        self.assertEqual(title, "architecture 1 layers ['Conv2D(32, (3, 3), padding=same)', 'MaxPooling2D((2, 2))', 'Flatten', 'Dense(256)', 'Dropout(0.25)', 'Dense(10)'] batchSize 512, epochs 5 valFraction 0.2")
        self.assertEqual(fileName, "test-figures/architecture-1-layers-conv2d32-3-3-paddingsame-maxpooling2d2-2-flatten-dense256-dropout025-dense10-batchsize-512-epochs-5-valfraction-02")

    def test_modelFunctioning_plottingMultipleRuns_withoutEvaluation(self):

        # when
        plotter: fnb.ModelRunsPlotter = architecture1.compilerDef(categoricalCompiler).fitterDef(fitterDef).trainSet(x_train, y_train)\
            .valFraction(0.2).fitModelAndRepeat(2).plotterDef(plotterDef)

        # then        
        title = plotter._title
        fileName = plotter.save()

        self.assertEqual(title, "architecture 1 layers ['Conv2D(32, (3, 3), padding=same)', 'MaxPooling2D((2, 2))', 'Flatten', 'Dense(256)', 'Dropout(0.25)', 'Dense(10)'] batchSize 512, epochs 5 valFraction 0.2")
        self.assertEqual(fileName, "test-figures/architecture-1-layers-conv2d32-3-3-paddingsame-maxpooling2d2-2-flatten-dense256-dropout025-dense10-batchsize-512-epochs-5-valfraction-02")

    def test_modelFunctioning_plottingMultipleRuns_withEvaluation(self):

        # when
        plotter: fnb.ModelRunsPlotter = architecture1.compilerDef(categoricalCompiler).fitterDef(fitterDef).trainSet(x_train, y_train)\
            .valFraction(0.2).fitModelAndRepeat(2).evalSet(x_test, y_test).evaluate().plotterDef(plotterDef)

        # then        
        title = plotter._title
        fileName = plotter.save()

        self.assertEqual(title, "architecture 1 layers ['Conv2D(32, (3, 3), padding=same)', 'MaxPooling2D((2, 2))', 'Flatten', 'Dense(256)', 'Dropout(0.25)', 'Dense(10)'] batchSize 512, epochs 5 valFraction 0.2")
        self.assertEqual(fileName, "test-figures/architecture-1-layers-conv2d32-3-3-paddingsame-maxpooling2d2-2-flatten-dense256-dropout025-dense10-batchsize-512-epochs-5-valfraction-02")




if __name__ == '__main__':
    unittest.main()