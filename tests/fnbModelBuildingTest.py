from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras import utils
import pandas as pd
import re

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

x_train = x_train_full[0:1000, :, :, :]
y_train = y_train_full[0:1000, :]
x_test = x_test_full[0:100, :, :, :]
y_test = y_test_full[0:100, :]

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

# categoricalCompiler = nna.compilerDef().loss('categorical_crossentropy').optimizer('adam').metrics(['accuracy'])

# oneEpochFitter = nna.fitterDef().batchSize(batchSize).epochs(1).verbose(True)

# validationDef = nna.validationDef().valFraction(0.2)

# plotterDef = nna.plotterDef()

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

def modelSummaryToDf(model: Sequential):
    summary = []
    model.summary(line_length = 120, print_fn = lambda s: summary.append(s))

    parsedSummary = []
    for i in range(0, len(summary)):
        res = re.search('\s+(.*) \((\w+)\)\s+\(([\w,\s]+)\)\s+(\d+)', summary[i])
        if res:
            lName = res.group(1)
            res1 = re.search('(\w+)_\d+', lName)
            if (res1):
                lName = res1.group(1)
            parsedSummary.append({'layerName': lName, 'layerType': res.group(2), 'outputShape': res.group(3), 'numParams': res.group(4)})
 
    df = pd.DataFrame(parsedSummary, columns= ['layerName', 'layerType', 'outputShape', 'numParams'])

    return df


print(modelSummaryToDf(refModel))

refModelSummaryDf = modelSummaryToDf(refModel)

class TestModelBuilding(unittest.TestCase):

    # def test0_standard_model_build():
        # model = Sequential()
        # model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape=(28, 28, 1)))
        # model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Flatten())
        # model.add(Dense(256, activation = 'relu'))
        # model.add(Dropout(0.25))
        # model.add(Dense(10, activation = 'softmax'))

        # model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        # model.summary()

        # history = model.fit(x_train, y_train, batch_size = batchSize, epochs = 1, verbose = 1)

        # scores = model.evaluate(x_test, y_test)
        

    def test_modelSummary(self):
        model: Sequential = architecture1.build((28, 28, 1))

        # model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        # history = model.fit(x_train, y_train, batch_size = batchSize, epochs = 1, verbose = 1)

        # summary = []
        # model.summary(print_fn = lambda s: summary.append(s))
        # model.summary(line_length = 120)

        # print("\n".join(refSummary))
        # self.assertEqual([refSummary[i] for i in range(1, len(refSummary))], [summary[i] for i in range(1, len(summary))])
        # self.assertEqual(len(refSummary), len(summary))
        
        modelSummaryDf = modelSummaryToDf(model)

        # print(refModelSummaryDf)
        # print(modelSummaryDf)
        
        self.assertTrue(refModelSummaryDf.equals(modelSummaryDf))

    def test_architectureTitle(self):
        
        # given
        categoricalCompiler = fnb.createCompilerDef().loss('categorical_crossentropy').optimizer('adam').metrics(['accuracy'])
        oneEpochFitter = fnb.createFitterDef().batchSize(batchSize).epochs(1).verbose(True)
        validationDef = fnb.createValidationDef().valFraction(0.2)

        # when
        archTitle = architecture1.composeTitle(categoricalCompiler, oneEpochFitter, validationDef)

        # then 
        self.assertEqual(archTitle, 
            "layers ['Conv2D(32, (3, 3), padding=same)', 'Conv2D(32, (3, 3), padding=same)', 'MaxPooling2D((2, 2))', 'Dropout(0.25)', 'Flatten', 'Dense(256)', 'Dropout(0.25)', 'Dense(10)'] "
                + "batchSize 512, epochs 1 valFraction 0.2")

    def test_architectureTitle_addingLayers(self):
        
        # given
        categoricalCompiler = fnb.createCompilerDef().loss('categorical_crossentropy').optimizer('adam').metrics(['accuracy'])
        oneEpochFitter = fnb.createFitterDef().batchSize(batchSize).epochs(1).verbose(True)
        validationDef = fnb.createValidationDef().valFraction(0.2)

        # when
        architecture2 = fnb.NnArch([], 'architecture2') \
            .add(fnb.layers().Dense().size(32)) \
            .add(fnb.layers().Flatten())
    
        # then 
        self.assertEqual(len(architecture2._layers), 2)
        
        archTitle = architecture2.composeTitle(categoricalCompiler, oneEpochFitter, validationDef)
        self.assertEqual(archTitle, 
            "layers ['Dense(32)', 'Flatten'] batchSize 512, epochs 1 valFraction 0.2")

    def test_modelFunctioning(self):
        # when
        model: Sequential = architecture1.build((28, 28, 1))

        # then
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        history = model.fit(x_train, y_train, batch_size = batchSize, epochs = nEpochs, verbose = 1)

        self.assertEqual(len(history.history['loss']), 5)

        scores = model.evaluate(x_test, y_test)        
        self.assertGreater(scores[0], 0.5)
        self.assertGreater(scores[1], 0.5)

    # def test2_architecture_build_compile():
    #     model: Sequential = architecture1.compiler(categoricalCompiler).compile((28, 28, 1))
    #     history = model.fit(x_train, y_train, batch_size = batchSize, epochs = 1, verbose = 1)

    # def test3_architecture_build_compile_fit_without_valSet():
    #     model = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).fitModel().ejectModel()
    #     model, history = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).fitModel().ejectModelAndHistory()

    # def test4_architecture_build_compile_fit_with_valSet():
    #     model = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).valSet(x_test, y_test).fitModel().ejectModel()
    #     model, history = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).valSet(x_test, y_test).fitModel().ejectModelAndHistory()

    # test0_standard_model_build()

    # test1_architecture_build()

    # test2_architecture_build_compile()

    # test3_architecture_build_compile_fit_without_valSet()

    # test4_architecture_build_compile_fit_with_valSet()

    # trainSetDef = trainSetDef().training(x_train, y_train).validation(x_test, y_test)
    # trainSetDef = trainSetDef().trainSet(x_train, y_train).validationFraction(0.2)

    # model, history, scores = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).valSet(x_test, y_test).fitModel().evalSet(x_test, y_test).evaluate()

    # model, history, scores = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).valFraction(0.2).fitModel().evalSet(x_test, y_test).evaluate()

    # model, history, scores = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).valFraction(0.2).fitModel().evalSet(x_test, y_test).evaluateAndPlot()

    # model, df = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).valFraction(0.2).runN(2).evalSet(x_test, y_test).run().ejectDataFrame()

    # model, df = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).valFraction(0.2).runN(2).evalSet(x_test, y_test).run().plotter(plotterDef).plot()

    # model, df = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).valFraction(0.2).runN(2).evalSet(x_test, y_test).run().plotter(plotterDef).save()

    # model, df = architecture1.compiler(categoricalCompiler).fitter(oneEpochFitter).trainSet(x_train, y_train).valFraction(0.2).runN(2).evalSet(x_test, y_test).run().plotter(plotterDef).saveAndPlot()


    # assert 2+3 == 6, 'wrong argument'

if __name__ == '__main__':
    unittest.main()