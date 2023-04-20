import numpy as np
import torch
from models.attention_aggregator import AttentionAggregator
from models.backbones.backbone_builder import BackboneBuilder
from torch import nn


class MILNetWithClinicalData(nn.Module):
    """Training with image and clinical data"""

    # RuntimeError: Adaptive pool MPS: If output is larger than input, output sizes must be multiples of input sizes
    # def __init__(self, num_classes, backbone_name, clinical_data_size=5, expand_times=10):
    def __init__(
        self, num_classes, backbone_name, clinical_data_size=23, expand_times=3
    ):
        super().__init__()

        print("training with image and clinical data")
        self.clinical_data_size = clinical_data_size
        self.expand_times = expand_times  # expanding clinical data to match image features in dimensions

        self.image_feature_extractor = BackboneBuilder(backbone_name)
        self.attention_aggregator = AttentionAggregator(
            self.image_feature_extractor.output_features_size, 1
        )  # inner_feature_size=1
        print(
            f"classifier/Linear({self.attention_aggregator.L + self.clinical_data_size * self.expand_times})"
        )
        # 256 + 23*expand_times
        self.classifier = nn.Sequential(
            nn.Linear(
                self.attention_aggregator.L
                + self.clinical_data_size * self.expand_times,
                64,
            ),
            nn.ReLU(),
            nn.Linear(64, num_classes),  #
        )

    def forward(self, bag_data, clinical_data):
        device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
        bag_data = torch.FloatTensor(bag_data.cpu().numpy()).squeeze(0).to(device)
        patch_features = self.image_feature_extractor(bag_data)
        aggregated_feature, attention = self.attention_aggregator(
            patch_features
        )  # [1,L],[1,N]
        fused_data = torch.cat(
            [aggregated_feature, clinical_data.repeat(1, self.expand_times).float()],
            dim=-1,
        )  # feature fusion
        result = self.classifier(fused_data)

        return result, attention