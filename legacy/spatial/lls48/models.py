import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


def pool_backbone_feature(feature):
    if feature.dim() == 4:
        feature = feature.permute(0, 3, 1, 2)
        feature = F.adaptive_avg_pool2d(feature, (1, 1)).flatten(1)
    return feature


class TripletModel(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()
        self.CFG = CFG
        if pretrained:
            print("Using default pre-trained weights for initialized fine-tuning model")
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if hasattr(self.model, "fc"):
            n_features = self.model.fc.in_features
            print(f"n_features: {n_features}")
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "head"):
            n_features = self.model.head.in_features
            print(f"n_features: {n_features}")
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")

        self.head = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        feature = pool_backbone_feature(self.model(x))
        output = self.head(feature)
        return output, feature


class TripletModel_with_MLP(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()
        self.CFG = CFG
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if hasattr(self.model, "fc"):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "head"):
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")

        self.head = SupConHead(n_features, 128)
        self.classification_head = nn.Linear(128, CFG.target_size)

    def forward(self, x):
        feature = pool_backbone_feature(self.model(x))
        projected = self.head(feature)
        logits = self.classification_head(projected)
        return logits, projected


class Model_feature_extractor(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()
        self.CFG = CFG
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if hasattr(self.model, "fc"):
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "head"):
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")

    def forward(self, x):
        return pool_backbone_feature(self.model(x))


class supcon_Model(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()
        self.CFG = CFG
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if hasattr(self.model, "fc"):
            n_features = self.model.fc.in_features
            print(f"n_features: {n_features}")
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "head"):
            n_features = self.model.head.in_features
            print(f"n_features: {n_features}")
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")

        self.head = nn.Sequential(
            nn.Linear(n_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128),
        )

    def forward(self, x):
        feature = pool_backbone_feature(self.model(x))
        output = self.head(feature)
        return output, feature


class SupConHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_dim),
        )

    def forward(self, x):
        return self.head(x)


class FeatureSupConModel(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()
        self.CFG = CFG
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if hasattr(self.model, "fc"):
            n_features = self.model.fc.in_features
            print(f"n_features: {n_features}")
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "head"):
            n_features = self.model.head.in_features
            print(f"n_features: {n_features}")
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")

        self.head = SupConHead(n_features, 128)

    def forward(self, x):
        feature = pool_backbone_feature(self.model(x))
        output = self.head(feature)
        return output
