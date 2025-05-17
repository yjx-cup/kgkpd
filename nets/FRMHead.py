import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRecalibrationModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureRecalibrationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.sigmoid(out)
        return out * identity


class FRMHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FRMHead, self).__init__()
        self.feature_recalibration = FeatureRecalibrationModule(in_channels)
        self.classification_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.bbox_regression_conv = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.offset_regression_conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, x):
        x = self.feature_recalibration(x)
        cls_preds = self.classification_conv(x)
        bbox_preds = self.bbox_regression_conv(x)
        offset_preds = self.offset_regression_conv(x)
        return cls_preds, bbox_preds, offset_preds


# Example usage
if __name__ == "__main__":
    frm_head = FRMHead(in_channels=256, num_classes=80)
    x = torch.randn(8, 256, 128, 128)  # Example input tensor
    cls_preds, bbox_preds, offset_preds = frm_head(x)
    print("Classification Predictions Shape:", cls_preds.shape)
    print("Bounding Box Predictions Shape:", bbox_preds.shape)
    print("Offset Box Predictions Shape:", offset_preds.shape)
