import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import fluid_nn_builder as fnb

import unittest

class TestLayers(unittest.TestCase):

    def test_Conv2D(self):
        layerBuilder = fnb.layers().Conv2D().kernel((3, 3)).size(32).padding('same').activation('relu')
        
        self.assertEqual(layerBuilder.describe(), 'Conv2D(32, (3, 3), padding=same)')

    def test_Dense(self):
        layerBuilder = fnb.layers().Dense().size(32).activation('relu')
        
        self.assertEqual(layerBuilder.describe(), 'Dense(32)')

    def test_MaxPooling2D(self):
        layerBuilder = fnb.layers().MaxPooling2D().kernel((3, 3))
        
        self.assertEqual(layerBuilder.describe(), 'MaxPooling2D((3, 3))')

    def test_Dropout(self):
        layerBuilder = fnb.layers().Dropout().rate(0.3)
        
        self.assertEqual(layerBuilder.describe(), 'Dropout(0.3)')

    def test_Flatten(self):
        layerBuilder = fnb.layers().Flatten()
        
        self.assertEqual(layerBuilder.describe(), 'Flatten')

    def test_BatchNormalization(self):
        layerBuilder = fnb.layers().BatchNormalization()
        
        self.assertEqual(layerBuilder.describe(), 'BatchNormalization')


if __name__ == '__main__':
    unittest.main()