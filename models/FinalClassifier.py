from torch import nn
import models


class TA3NClassifier(models.VideoModel):
    def __init__(self, num_class, modality, model_config, **kwargs):
        super().__init__(num_class=num_class, model_config=model_config)
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

        self.fc1 = nn.Linear(model_config["input_size"], model_config["hidden_size"])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=model_config["dropout"])
        self.fc2 = nn.Linear(model_config["hidden_size"], num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, {}

    def get_augmentation(self, modality):
        assert False
        if modality == 'RGB':
            train_augmentation = torchvision.transforms.Compose(
                # Data augmentation, at first reduce then interpolate
                [GroupMultiScaleCrop(self.model_config.resolution, [1, .875, .75]),
                 GroupRandomHorizontalFlip(is_flow=False),
                 Stack(roll=False),
                 ToTorchFormatTensor(div=not self.model_config.normalize),
                 GroupNormalize(self.model_config.normalize, self.input_mean, self.input_std, self.range)]
            )

            val_augmentation = torchvision.transforms.Compose([
                GroupCenterCrop(self.model_config.resolution),
                Stack(roll=False),
                ToTorchFormatTensor(div=not self.model_config.normalize),
                GroupNormalize(self.model_config.normalize, self.input_mean, self.input_std, self.range)
            ])
        else:
            raise NotImplementedError

        return train_augmentation, val_augmentation