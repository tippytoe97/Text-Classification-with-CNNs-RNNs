import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, w2vmodel, num_classes, window_sizes=(1,2,3,5)):
        '''
        Initialize CNN with the embedding layer, convolutional layer and a linear layer. 
        Steps to initialize -
        1. For embedding layer, create embedding from pretrained model using w2vmodel weights vectors, and padding.
        nn.Embedding would be useful.

        2. Create convolutional layers with the given window_sizes and padding of (window_size - 1, 0).
        nn.Conv2d would be useful. Additionally, nn.ModuleList might also be useful.

        3. Linear layer with num_classes output.

        Args:
            w2vmodel: Pre-trained word2vec model.
            num_classes: Number of classes (labels).
            window_sizes: Window size for the convolution kernel.
        '''
        super(CNN, self).__init__()
        weights = w2vmodel.wv # use this to initialize the embedding layer
        EMBEDDING_SIZE = 500  # Note that this is the embedding dimension in the word2vec model passed in as w2vmodel.wv
        NUM_FILTERS = 10      # Number of filters in CNN


        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2vmodel.wv.vectors), freeze=True, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, 
                out_channels=NUM_FILTERS, 
                kernel_size=(window_size, EMBEDDING_SIZE), 
                padding=(window_size - 1, 0)
            )
                
            for window_size in window_sizes])

        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)

    def forward(self, x):
        '''
        Implement the forward function to feed the input through the model and get the output.

        1. Feed the input through the embedding layer. This step should be implemented in forward_embed().
        2. Feed the result through the convolution layers. This step should be implemented in forward_convs().
        3. Feed the output from convolution through a fully connected layer to get the logits. You may need to manipulate
           the convolution output in order to feed it into a linear layer. 

        Args:
            x: Input data. A tensor of size (B, T) containing the input sequences, where B = batch size and T = sequence length

        Returns:
            output: Logits of each label. A tensor of size (B, C) where B = batch size and C = num_classes
        '''
        embed = self.forward_embed(x)     # (B, T, E)
        conv = self.forward_convs(embed) 
        output = self.fc(conv)
        return output

    def forward_embed(self, x):
        '''
        Pass the input through the embedding layer.

        Args: 
            x : A tensor of shape (B, T) containing the input sequences 

        Returns: 
            embeddings : A (B, T, E) tensor containing the embeddings corresponding to the input sequences, where E = embedding size.

        '''
        embeddings = self.embedding(x)
        return embeddings

    def forward_convs(self, embed):
        '''
        Pass the result of the embedding function through the convolution layers. For convolution layers, pass the convolution output through
        tanh and max_pool1d. 
        Args:
            embed: A tensor of size (B, T, E) containing the embeddings corresponding to the original input sequences.
        Returns:
            output: A tensor of size (B, F, K) where F = number of filters and K = len(window_sizes)

        '''
        embed = embed.unsqueeze(1)  # (B, 1, T, E)
        conv_outputs = []

        for conv in self.convs:
            x = F.relu(conv(embed))         # (B, F, T_out, 1)
            x = x.squeeze(3)                    # (B, F, T_out)
            x = F.max_pool1d(x, kernel_size=x.shape[2])  # (B, F, 1)
            x = x.squeeze(2)                    # (B, F)
            conv_outputs.append(x)

        output = torch.cat(conv_outputs, dim=2)  # (B, F * len(window_sizes))
        return output

