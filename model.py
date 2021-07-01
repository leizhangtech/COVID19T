import torch
import torch.nn as nn
from torchvision import models
import timm


class COVNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        model = models.resnet152(pretrained=True)
        layer_list = list(model.children())[:-2]
        self.pretrained_model = nn.Sequential(*layer_list)

        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(2048, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output


# resnet152 larger network COVNetL
class COVNetL(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        model = models.resnet152(pretrained=True)
        layer_list = list(model.children())[:-1]
        self.pretrained_model = nn.Sequential(*layer_list)

        # self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(2048, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        pooled_features = self.pretrained_model(x)
       # pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output

#COVNetT transformer
class COVNetT(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        layer_list = list(model.children())[:-2]


        self.pretrained_model = nn.Sequential(*layer_list)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(1024, n_classes)
        self.n_classes = n_classes

    def reshape_transform(tensor, height=7, width=7):
        result = tensor.reshape(tensor.size(0),
                                height, width, tensor.size(2))
    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model(x)
        features = torch.transpose(features, 2, 1)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.head(flattened_features)
        return output

#  BigThansfer (BiT)   |o k| resnetv2_101x1_bitm_in21k |x out of memory|resnetv2_50x3_bitm_in21k
class COVNetBiT(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        model = timm.create_model('resnetv2_101x1_bitm_in21k', pretrained=True)
        layer_list = list(model.children())[:-2]
        self.pretrained_model = nn.Sequential(*layer_list)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(2048, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output


#  Efficientnetv2   |o k| resnetv2_101x1_bitm_in21k |x out of memory|resnetv2_50x3_bitm_in21k
class COVNetEffi(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        model = timm.create_model('tf_efficientnetv2_m_in21k', pretrained=True)
        layer_list = list(model.children())[:-2]
        self.pretrained_model = nn.Sequential(*layer_list)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(1280, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output