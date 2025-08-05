import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab, num_classes):
        '''
        Initialize RNN with the embedding layer, bidirectional RNN layer and a linear layer with a dropout.
    
        Args:
        vocab: Vocabulary.
        num_classes: Number of classes (labels).
        
        '''
        super(RNN, self).__init__()
        self.embed_len = 50  # embedding_dim default value for embedding layer
        self.hidden_dim = 75 # hidden_dim default value for rnn layer
        self.n_layers = 1    # num_layers default value for rnn
        self.p = 0.5   # default value for the dropout probability, you may change this

        self.embedding = nn.Embedding(len(vocab),  self.embed_len)
        self.rnn = nn.RNN(self.embed_len, self.hidden_dim, batch_first=True, bidirectional=True) 
        self.linear = nn.Linear(self.hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(self.p)

    def forward(self, inputs, inputs_len):
        '''
        Implement the forward function to feed the input through the model and get the output.

        1. Pass the input sequences through the embedding layer to obtain the embeddings. This step should be implemented in forward_embed().
        2. Pass the embeddings through the rnn layer to obtain the output. This step should be implemented in forward_rnn().
        3. Concatenate the hidden states of the rnn as shown in the architecture diagram in HW3.ipynb. This step should be implemented in forward_concat().
        4. Pass the output from step 3 through the linear layer.

        Args:
            inputs : A (B, L) tensor containing the input sequences, where B = batch size and L = sequence length
            inputs_len :  A (B, ) tensor containing the lengths of the input sequences in the current batch prior to padding.

        Returns:
            output: Logits of each label. A tensor of size (B, C) where B = batch size and C = num_classes
        '''
        embed = self.forward_embed(inputs)           # (B, L, E)
        rnn_output = self.forward_rnn(embed, inputs_len)  # (L, B, 2H)
        concat = self.forward_concat(rnn_output, inputs_len)     # (B, 2H)
        output = self.linear(self.dropout(concat))   # (B, C)
        return output

    def forward_embed(self, inputs):
        """
        Pass the input sequences through the embedding layer.

        Args: 
            inputs : A (B, L) tensor containing the input sequences

        Returns: 
            embeddings : A (B, L, E) tensor containing the embeddings corresponding to the input sequences, where E = embedding length.
        """

        embeddings = self.embedding(inputs)
        return embeddings
    
    def forward_rnn(self, embeddings, inputs_len):
        """
        Pack the input sequence embeddings, and then pass it through the RNN layer to get the output from the RNN layer, which should be padded.

        Args: 
            embeddings : A (B, L, E) tensor containing the embeddings corresponding to the input sequences.
            inputs_len : A (B, ) tensor containing the lengths of the input sequences prior to padding.

        Returns: 
            output : A (B, L', 2 * H) tensor containing the output of the RNN. L' = the max sequence length in the batch (prior to padding) = max(inputs_len), and H = the hidden embedding size.

        """
    
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, inputs_len.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through the RNN
        packed_output, hidden = self.rnn(packed)

        # Pad the packed sequence back to (B, L', 2 * H)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        return output
    
    def forward_concat(self, rnn_output, inputs_len):
        """
        Concatenate the first hidden state in the reverse direction and the last hidden state in the forward direction of the bidirectional RNN. 
        Take a look at the architecture diagram of our model in HW3.ipynb to visually see how this is done. Also, keep in mind the important note
        below the architecture diagram.

        Args: 
            rnn_output : A (B, L', 2 * H) tensor containing the output of the RNN.
            inputs_len : A (B, ) tensor containing the lengths of the input sequences prior to padding.

        Returns: 
            concat : A (B, 2 * H) tensor containing the two hidden states concatenated together.
        """

        B = rnn_output.size(0)
        H = rnn_output.size(2) // 2  # since output is (B, L, 2*H)
        # Forward direction: select last valid timestep for each sample
        forward_indices = (inputs_len - 1).view(-1, 1, 1).expand(-1, 1, H)  # (B, 1, H)
        forward_hidden = rnn_output.gather(1, forward_indices).squeeze(1)  # (B, H)

        # Reverse direction: always at time step 0 (stored after forward)
        reverse_hidden = rnn_output[:, 0, H:]  # (B, H)

        # Concatenate
        concat = torch.cat((forward_hidden, reverse_hidden), dim=1)  # (B, 2H)
        return concat

        