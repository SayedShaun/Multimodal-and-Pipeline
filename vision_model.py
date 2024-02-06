import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.vgg = models.vgg16(init_weights="imagenet")
        self.vgg.classifier[-1] = nn.Linear(4096, embed_size)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = F.dropout(self.vgg(images), p=0.5)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.lstm_layer = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.3)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features, captions):
        image_features = image_features.unsqueeze(0)
        embedding = F.dropout(self.embedding_layer(captions), p=0.3)
        concat = torch.cat((image_features, embedding), dim=0)
        lstm_hidden, _ = self.lstm_layer(concat)
        output = self.linear(lstm_hidden)
        return output


class VisionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(VisionModel, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs