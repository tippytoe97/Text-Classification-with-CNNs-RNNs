import numpy as np
import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn


class Word2Vec(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocabulary_size = 0

    def tokenize(self, data):
        """
        Split all the words in the data into tokens.

        Args:
            data: (N,) list of sentences in the dataset.

        Return:
            tokens: (N, D_i) list of tokenized sentences. The i-th sentence has D_i words after the split.
        """
        tokens = [sentence.split() for sentence in data]
        return tokens


    def create_vocabulary(self, tokenized_data):
        """
        Create a vocabulary for the tokenized data.
        For each unique word in the vocabulary, assign a unique ID to the word. 
        Assign the word to ID mapping to word2idx variable.
        Assign the ID to word mapping to idx2word variable.
        Assign the size of the vocabulary to vocabulary_size variable.

        Args:
            tokenized_data: (N, D) list of split tokens in each sentence.
            
        Return:
            None (The update is done for self.word2idx, self.idx2word and self.vocabulary_size)
        """
         # Flatten all tokens into a single list
        all_tokens = [word for sentence in tokenized_data for word in sentence]

        # Get unique words and sort them alphabetically
        unique_words = sorted(set(all_tokens))

        # Create word-to-index and index-to-word mappings
        self.word2idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Set vocabulary size
        self.vocabulary_size = len(unique_words)


    def skipgram_embeddings(self, tokenized_data, window_size=2):
        """
        Create a skipgram embeddings by taking context as middle word and predicting
        N=window_size past words and N=window_size future words.

        NOTE : In case the window range is out of the sentence length, create a
        context by feasible window size. For example : The sentence with tokenIds
        as follows will be represented as
        [1, 2, 3, 4, 5] ->
           source_tokens             target_tokens
           [1]                       [2]
           [1]                       [3]
           [2]                       [1]
           [2]                       [3]
           [2]                       [4]
           [3]                       [1]
           [3]                       [2]
           [3]                       [4]
           [3]                       [5]
           [4]                       [2]
           [4]                       [3]
           [4]                       [5]
           [5]                       [3]
           [5]                       [4]

        source_tokens: [[1], [1], [2], [2], [2], ...]
        target_tokens: [[2], [3], [1], [3], [4], ...]
        Args:
            tokenized_data: (N, D_i) list of split tokens in each sentence.
            window_size: length of the window for creating context. Default is 2.

        Returns:
            source_tokens: List of elements where each element is the middle word in the window.
            target_tokens: List of elements representing IDs of the context words.

        NOTE: We are returning token IDs (ints), not the actual tokens (strings).
        """
        source_tokens = []
        target_tokens = []

        for sentence in tokenized_data:
            # Convert words to their corresponding IDs
            token_ids = [self.word2idx[word] for word in sentence]

            for i in range(len(token_ids)):
                center = token_ids[i]
                # Window range: max(0, i - window_size) to min(len, i + window_size + 1)
                start = max(0, i - window_size)
                end = min(len(token_ids), i + window_size + 1)

                for j in range(start, end):
                    if j != i:
                        source_tokens.append([center])
                        target_tokens.append([token_ids[j]])

        return source_tokens, target_tokens
    

    def cbow_embeddings(self, tokenized_data, window_size=2):
        """
        Create a cbow embeddings by taking context as N=window_size past words and N=window_size future words.

        NOTE : In case the window range is out of the sentence length, create a
        context by feasible window size. For example : The sentence with tokenIds
        as follows will be represented as
        [1, 2, 3, 4, 5] ->
           source_tokens             target_tokens
           [2,3]                     [1]
           [1,3,4]                   [2]
           [1,2,4,5]                 [3]
           [2,3,5]                   [4]
           [3,4]                     [5]
           
        source_tokens: [[2,3], [1,3,4], [1,2,4,5], [2,3,5], [3,4]]
        target_tokens: [[1], [2], [3], [4], [5]]

        Args:
            tokenized_data: (N, D_i) list of split tokens in each sentence.
            window_size: length of the window for creating context. Default is 2.

        Returns:
            source_tokens: List of elements where each element is maximum of N=window_size*2 context word IDs.
            target_tokens: List of elements representing IDs of the middle word in the window.

        NOTE: We are returning token IDs (ints), not the actual tokens (strings).
        """
        source_tokens = []
        target_tokens = []

        for sentence in tokenized_data:
            # Convert words to their corresponding IDs
            token_ids = [self.word2idx[word] for word in sentence]

            for i in range(len(token_ids)):
                
                start = max(0, i - window_size)
                end = min(len(token_ids), i + window_size + 1)

                context = [token_ids[j] for j in range(start, end) if j != i]

                if context:
                    source_tokens.append(context)
                    target_tokens.append([token_ids[i]])

        return source_tokens, target_tokens


class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        """
        Initialize SkipGram_Model with the embedding layer and a linear layer.
        
        Reference: 
            embedding - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            linear layer - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        Args:
            vocab_size: Size of the vocabulary.
        """

        super(SkipGram_Model, self).__init__()
        self.EMBED_DIMENSION = 300 
        self.EMBED_MAX_NORM = 1    


        self.embeddings = torch.nn.Embedding(vocab_size , self.EMBED_DIMENSION, max_norm = self.EMBED_MAX_NORM)
        self.linear = torch.nn.Linear(self.EMBED_DIMENSION, vocab_size)


    def forward(self, inputs):
        """
        Implement the SkipGram model architecture

        Args:
            inputs: (B, 1) Tensor of IDs for each sentence. Where B = batch size

        Returns:
            output: (B, V) Tensor of logits where V = vocab_size.
            
            No need to have a softmax layer here.
        """

        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
            
        embedded = self.embeddings(inputs)  # shape (B, 300)
        output = self.linear(embedded)

        return output


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        """
        Initialize CBOW_Model with the embedding layer and a linear layer.
        
        Reference: 
            embedding - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            linear layer - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        Args:
            vocab_size: Size of the vocabulary.
        """
        super(CBOW_Model, self).__init__()
        self.EMBED_DIMENSION = 300  
        self.EMBED_MAX_NORM = 1  


        self.embeddings = torch.nn.Embedding(vocab_size, self.EMBED_DIMENSION, max_norm = self.EMBED_MAX_NORM)
        self.linear =  torch.nn.Linear(self.EMBED_DIMENSION, vocab_size)

    def forward(self, inputs):
        """

        Args:
            inputs: (D_i) Tensor of IDs for each sentence.

        Returns:
            output: (1, V) Tensor of logits
            
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        embedded = self.embeddings(inputs)
        mean = embedded.mean(dim=1)

        output = self.linear(mean)

        return output
