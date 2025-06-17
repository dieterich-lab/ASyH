from os.path import dirname, join
import pytest

import ASyH.models
import ASyH.data


_input_data_cache = None


@pytest.fixture
def input_data():
    global _input_data_cache
    if _input_data_cache is None:
        input_file = join(dirname(dirname(__file__)), 'examples', 'Kaggle_Sirio_Libanes-16features.xlsx')
        _input_data_cache = ASyH.data.Data()
        _input_data_cache.read(input_file)
        metadata_file = join(dirname(dirname(__file__)), 'examples', 'Kaggle_Sirio_Libanes-16features.json')
        _metadata = ASyH.metadata.Metadata()
        _metadata.read(metadata_file)
        _input_data_cache.set_metadata(_metadata)
    yield _input_data_cache


def test_construct_tvae_model():
    """Testing initialization of the TVAE model"""
    m = ASyH.models.TVAEModel()
    assert m.model_type == 'TVAESynthesizer'
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
    assert m.model_type == 'CTGANSynthesizer'
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
    assert m.model_type == 'CopulaGANSynthesizer'
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
    assert m.model_type == 'GaussianCopulaSynthesizer'
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


def test_construct_forest_flow_model():
    """Testing initialization of the CTGAN model"""
    m = ASyH.models.ForestFlowModel()
    assert m.model_type == 'ForestFlowSynthesizer'
    # private state:
    assert m._trained is False
    assert m._training_data is None
    assert m._metadata is None


def test_train_synthesize_forest_flow_model(input_data):
    """Testing training and synthesis with CTGANModel.  Since the training
    process takes a long time, training and synthesis are combined into one
    test.
    """
    m = ASyH.models.ForestFlowModel()
    # without specifying any training data in ctor or _train(), there should
    # be an AssertError in _train():
    with pytest.raises(AssertionError):
        m._train()
    m._train(input_data)
    assert m._trained
    m.synthesize()


def test_construct_cpar_model():
    """Testing initialization of the CPAR model"""
    m = ASyH.models.CPARModel()
    assert m.model_type == 'CPARSynthesizer'
    # private state:
    assert m._trained is False
    assert m._training_data is None
    assert m._metadata is None


def test_train_synthesize_cpar_model(input_data):
    """Testing training and synthesis with CPAR model.  Since the training
    process takes a long time, training and synthesis are combined into one
    test.
    """
    m = ASyH.models.CPARModel()
    # without specifying any training data in ctor or _train(), there should
    # be an AssertError in _train():
    with pytest.raises(AssertionError):
        m._train()
    m._train(input_data)
    assert m._trained
    m.synthesize()