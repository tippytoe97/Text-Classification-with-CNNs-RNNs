import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, feat_size, num_classes):
        '''
        Initialize the neural net with the linear layers.
    
        Args:
            feat_size: number of features of one input datapoint
            num_classes: Number of classes (labels)    
        '''
        super(NN, self).__init__()
        hidden_size = 46  # You can tune this number

        # Two required linear layers
        self.fc1 = nn.Linear(feat_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        '''
        Implement the forward function to feed the input x through the model and get the output.

        Args:
            x: (B, D) tensor containing the encoded data where B = batch size and D = feat_size

        Returns:
            output: (B, C) tensor of logits where B = batch size and C = num_classes

        '''
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output
