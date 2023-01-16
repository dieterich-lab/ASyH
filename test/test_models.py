from os.path import dirname, join
import pytest

import ASyH.models
import ASyH.data


@pytest.fixture
def input_data():
    input_file = join(dirname(__file__), 'testdata.csv')
    data = ASyH.data.Data()
    data.read(input_file)
    yield data


def test_construct_tvae_model():
    """Testing initialization of the TVAE model"""
    m = ASyH.models.TVAEModel()
    assert m.model_type == 'TVAE'
    # private state:
    assert m._trained is False
    assert m._training_data is None
    assert m._metadata is None


def test_train_synthesize_tvae_model(input_data):
    """Testing training and synthesis with TVAEModel.  Since the training
    process takes a long time, training and synthesis are combined into one
    test.
    """
    m = ASyH.models.TVAEModel()
    # without specifying any training data in ctor or _train(), there should
    # be an AssertError in _train():
    with pytest.raises(AssertionError):
        m._train()
    m._train(input_data)
    assert m._trained
    m.synthesize()


def test_construct_ctgan_model():
    """Testing initialization of the CTGAN model"""
    m = ASyH.models.CTGANModel()
    assert m.model_type == 'CTGAN'
    # private state:
    assert m._trained is False
    assert m._training_data is None
    assert m._metadata is None


def test_train_synthesize_ctgan_model(input_data):
    """Testing training and synthesis with CTGANModel.  Since the training
    process takes a long time, training and synthesis are combined into one
    test.
    """
    m = ASyH.models.CTGANModel()
    # without specifying any training data in ctor or _train(), there should
    # be an AssertError in _train():
    with pytest.raises(AssertionError):
        m._train()
    m._train(input_data)
    assert m._trained
    m.synthesize()


def test_construct_copula_gan_model():
    '''Testing initialization of the CopulaGAN model'''
    m = ASyH.models.CopulaGANModel()
    assert m.model_type == 'CopulaGAN'
    # private state:
    assert m._trained is False
    assert m._training_data is None
    assert m._metadata is None


def test_train_synthesize_copula_gan_model(input_data):
    """Testing training and synthesis with CopulaGANModel.  Since the training
    process takes a long time, training and synthesis are combined into one
    test.
    """
    m = ASyH.models.TVAEModel()
    # without specifying any training data in ctor or _train(), there should
    # be an AssertError in _train():
    with pytest.raises(AssertionError):
        m._train()
    m._train(input_data)
    assert m._trained
    m.synthesize()


def test_construct_gaussian_copula_model():
    """Testing initialization of the GaussianCopula model"""
    m = ASyH.models.GaussianCopulaModel()
    assert m.model_type == 'GaussianCopula'
    # private state:
    assert m._trained is False
    assert m._training_data is None
    assert m._metadata is None


def test_train_synthesize_gaussian_copula_model(input_data):
    """Testing training and synthesis with GaussianCopulaModel.  Since the training
    process takes a long time, training and synthesis are combined into one
    test.
    """
    m = ASyH.models.GaussianCopulaModel()
    # without specifying any training data in ctor or _train(), there should
    # be an AssertError in _train():
    with pytest.raises(AssertionError):
        m._train()
    m._train(input_data)
    assert m._trained
    m.synthesize()
