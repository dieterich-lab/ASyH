import unittest

import ASyH.models
import ASyH.data


class TestASyHModel(unittest.TestCase):

    def setUp(self):
        # a reduced dataset derived from Kaggle: Sirio-Libanes Covid data
        self.testdata = ASyH.data.Data()
        self.testdata.read("testdata.csv")

    def test_construct_TVAEModel(self):
        '''Testing initialization of the TVAE model'''
        m = ASyH.models.TVAEModel()
        self.assertEqual(m.model_type, 'TVAE')
        # private state:
        self.assertFalse(m._trained)
        self.assertFalse(m._training_data)
        self.assertFalse(m._metadata)

    def test_train_synthesize_TVAEModel(self):
        '''Testing training and synthesis with TVAEModel.  Since the training
        process takes a long time, training and synthesis are combined into one
        test.
        '''
        m = ASyH.models.TVAEModel()
        # without specifying any training data in ctor or _train(), there should
        # be an AssertError in _train():
        with self.assertRaises(AssertionError):
            m._train()
        m._train(self.testdata)
        self.assertTrue(m._trained)
        sample = m.synthesize()

    def test_construct_CTGANModel(self):
        '''Testing initialization of the CTGAN model'''
        m = ASyH.models.CTGANModel()
        self.assertEqual(m.model_type, 'CTGAN')
        # private state:
        self.assertFalse(m._trained)
        self.assertFalse(m._training_data)
        self.assertFalse(m._metadata)

    def test_train_synthesize_CTGANModel(self):
        '''Testing training and synthesis with CTGANModel.  Since the training
        process takes a long time, training and synthesis are combined into one
        test.
        '''
        m = ASyH.models.CTGANModel()
        # without specifying any training data in ctor or _train(), there should
        # be an AssertError in _train():
        with self.assertRaises(AssertionError):
            m._train()
        m._train(self.testdata)
        self.assertTrue(m._trained)
        sample = m.synthesize()

    def test_construct_CopulaGANModel(self):
        '''Testing initialization of the CopulaGAN model'''
        m = ASyH.models.CopulaGANModel()
        self.assertEqual(m.model_type, 'CopulaGAN')
        # private state:
        self.assertFalse(m._trained)
        self.assertFalse(m._training_data)
        self.assertFalse(m._metadata)

    def test_train_synthesize_CopulaGANModel(self):
        '''Testing training and synthesis with CopulaGANModel.  Since the training
        process takes a long time, training and synthesis are combined into one
        test.
        '''
        m = ASyH.models.TVAEModel()
        # without specifying any training data in ctor or _train(), there should
        # be an AssertError in _train():
        with self.assertRaises(AssertionError):
            m._train()
        m._train(self.testdata)
        self.assertTrue(m._trained)
        sample = m.synthesize()

    def test_construct_GaussianCopulaModel(self):
        '''Testing initialization of the GaussianCopula model'''
        m = ASyH.models.GaussianCopulaModel()
        self.assertEqual(m.model_type, 'GaussianCopula')
        # private state:
        self.assertFalse(m._trained)
        self.assertFalse(m._training_data)
        self.assertFalse(m._metadata)

    def test_train_synthesize_GaussianCopulaModel(self):
        '''Testing training and synthesis with GaussianCopulaModel.  Since the training
        process takes a long time, training and synthesis are combined into one
        test.
        '''
        m = ASyH.models.GaussianCopulaModel()
        # without specifying any training data in ctor or _train(), there should
        # be an AssertError in _train():
        with self.assertRaises(AssertionError):
            m._train()
        m._train(self.testdata)
        self.assertTrue(m._trained)
        sample = m.synthesize()


if __name__ == '__main__':
    unittest.main()
