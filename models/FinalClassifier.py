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
        self.model = TA3N(num_class, baseline_type=None, frame_aggregation="avgpool", modality=modality, base_model="I3D", partial_bn=False)

    def forward(self, x):
        x = self.model(*x)
        assert False, x
        return x, {}

    def get_augmentation(self, modality):
        return super().get_augmentation(modality)
