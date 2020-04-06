import core
import gin


rescale_preprocess = gin.external_configurable(
    core.rescale_preprocess, blacklist=['image', 'labels'])
get_datasets = gin.external_configurable(core.get_datasets)
cnn_features = gin.external_configurable(
    core.cnn_features, blacklist=['image'])
mlp_features = gin.external_configurable(
    core.mlp_features, blacklist=['inp'])
get_classification_model = gin.external_configurable(
    core.get_classification_model, blacklist=['input_spec', 'num_classes'])
fit = gin.external_configurable(
    core.fit, blacklist=['model', 'train_ds', 'test_ds'])


config = '''
get_datasets.train_map_fn=@rescale_preprocess
get_datasets.test_map_fn=@rescale_preprocess
cnn_features.Conv2D = @Conv2D
cnn_features.Dense = @Dense
mlp_features.Dense = @Dense
get_classification_model.features_fn = @cnn_features
get_classification_model.Dense = @Dense
'''

gin.parse_config(config)
