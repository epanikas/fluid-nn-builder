import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import fluid_nn_builder as fnb

import unittest

layers = [
    fnb.layers().Conv2D().size(32).kernel((3, 3)).padding('same').activation('relu'),
    fnb.layers().MaxPooling2D().kernel((3, 3)),
    fnb.layers().Flatten(),
    fnb.layers().BatchNormalization(),
    fnb.layers().Dense().size(256).activation('relu'),
    fnb.layers().Dropout().rate(0.25),
    fnb.layers().Dense().size(10).activation('softmax'),
]

x_train = [1, 2, 3, 4]
y_train = [4, 5, 6, 7]

x_test = [1, 2]
y_test = [4, 5]

class TestDefinitions(unittest.TestCase):

    def test_compilerDef(self):

        compilerDef = fnb.createCompilerDef().loss('categorical_crossentropy').optimizer('adam').metrics(['accuracy'])

        self.assertEqual(compilerDef.describe(), "loss categorical_crossentropy, optimizer adam, metrics ['accuracy']")

    def test_fitterDef(self):
        
        fitterDef = fnb.createFitterDef().batchSize(128).epochs(1).verbose(True)
        
        self.assertEqual(fitterDef.describe(), 'batchSize 128, epochs 1')

    def test_validationDef_valFraction(self):
        
        validationDef = fnb.createValidationDef().valFraction(0.25)
        
        self.assertEqual(validationDef.describe(), 'valFraction 0.25')

    def test_validationDef_valSet(self):
        
        validationDef = fnb.createValidationDef().valSet(x_test, y_test)
        
        self.assertEqual(validationDef.describe(), 'valSet size 2')

    def test_plotterDef(self):
        
        compilerDef = fnb.createCompilerDef().loss('categorical_crossentropy').optimizer('adam').metrics(['accuracy'])
        fitterDef = fnb.createFitterDef().batchSize(128).epochs(1).verbose(True)
        validationDef = fnb.createValidationDef().valFraction(0.25)
        plotterDef = fnb.createPlotterDef().withArchName(True).withCompilerDef(True).withFitterDef(True).withLayers(True)
        
        self.assertEqual(plotterDef.composeTitle(layers, compilerDef, fitterDef, validationDef, 'name1'), 
        "name1 layers ['Conv2D(32, (3, 3), padding=same)', 'MaxPooling2D((3, 3))', 'Flatten', 'BatchNormalization', 'Dense(256)', 'Dropout(0.25)', 'Dense(10)'] "
            + "loss categorical_crossentropy, optimizer adam, metrics ['accuracy'] batchSize 128, epochs 1 valFraction 0.25")


if __name__ == '__main__':
    unittest.main()