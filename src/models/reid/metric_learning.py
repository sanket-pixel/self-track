import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.nn import functional as F

class VisualEmbedding(nn.Module):

    def __init__(self):
        super(VisualEmbedding, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        self.resnet50 = nn.Sequential(*(list(resnet50.children())[:-1]))


    def forward(self, x):
        x = self.resnet50(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        return x

class ProjectionLayer(nn.Module):

    def __init__(self, input_dim,output_dim):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(input_dim,output_dim)

    def forward(self, x):
        x = self.projection(x)
        return x


class ReIDModel(nn.Module):

    def __init__(self, embedding_dim, projection_dim):
        super(ReIDModel, self).__init__()
        self.visual_embedder = VisualEmbedding()
        self.projection_layer = ProjectionLayer(embedding_dim, projection_dim)

    def forward(self, patch, mode = "train"):
        if mode=="train":
            visual_emb = F.normalize(self.visual_embedder(patch),dim=1)
            projection_emb = F.normalize(self.projection_layer(visual_emb),dim=1)
            return projection_emb
        if mode=="eval":
            visual_emb = F.normalize(self.visual_embedder(patch),dim=1)
            return visual_emb



