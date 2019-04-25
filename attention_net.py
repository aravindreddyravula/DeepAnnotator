import torch
from torch import nn
import utils
import torch.nn.functional as F
import Config

class Attention_Net(nn.Module):
    def __init__(self, sent_size):
        super(Attention_Net, self).__init__()
        self.embeds = nn.Embedding(len(utils.create_vocabulary(Config.window_size)), Config.embedding_size)
        self.embeds_size = Config.embedding_size * sent_size
        self.tanh = torch.tanh
        self.fc1 = nn.Linear(self.embeds_size, Config.hidden_layer_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = F.relu
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(Config.hidden_layer_size, 1)

    def forward(self, inputs):
        embedding_weights = self.embeds(inputs).view((-1, self.embeds_size))
        attended_inputs = embedding_weights
        layer1 = self.fc1(attended_inputs)
        layer1 = self.dropout(layer1)
        act1 = self.relu(layer1)
        layer2 = self.fc2(act1)
        layer2 = self.dropout(layer2)
        act2 = self.relu(layer2)
        output = self.sigmoid(act2)
        return output