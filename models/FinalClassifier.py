from torch import nn
import models
from models.TA3N import TA3N


class TA3NClassifier(models.VideoModel):
    def __init__(self, num_class, modality, model_config, **kwargs):
        super().__init__(num_class=num_class, model_config=model_config)
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

        self.num_class = num_class
        self.model = TA3N(num_class, baseline_type=None, frame_aggregation=model_config.temporal_aggregation,
                          modality=modality, base_model="I3D", partial_bn=False)

        self.beta = model_config.beta
        self.mu = model_config.mu
        self.reverse = model_config.reverse

    def forward(self, x):
        source_data, target_data, is_training = x

        x = self.model(source_data, target_data, self.beta, self.mu, is_training, self.reverse)
        assert False, x[0]
        return x, {}

    def get_augmentation(self, modality):
        return super().get_augmentation(modality)
