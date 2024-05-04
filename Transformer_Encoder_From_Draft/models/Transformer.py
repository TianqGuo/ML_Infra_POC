

import numpy as np

import torch
from torch import nn
import random

class TransformerTranslator(nn.Module):
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)

        self.embeddingL = nn.Embedding(self.input_size, self.hidden_dim)
        self.posembeddingL = nn.Embedding(self.max_length, self.hidden_dim)

        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.hidden_dim)
        )
        self.norm_ff = nn.LayerNorm(self.hidden_dim)

        self.fl = nn.Linear(self.hidden_dim, self.output_size)

        
    def forward(self, inputs):
        """
        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """
        embeddings = self.embed(inputs)

        attention_out = self.multi_head_attention(embeddings)

        ff_out = self.feedforward_layer(attention_out)

        outputs = self.final_layer(ff_out)

        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """

        word_embeddings = self.embeddingL(inputs)

        positions = torch.arange(inputs.size(1), device=inputs.device).expand_as(inputs)

        position_embeddings = self.posembeddingL(positions)

        embeddings = word_embeddings + position_embeddings

        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """

        k1, v1, q1 = self.k1(inputs), self.v1(inputs), self.q1(inputs)
        k2, v2, q2 = self.k2(inputs), self.v2(inputs), self.q2(inputs)

        head1_output = self.softmax(q1 @ k1.transpose(-2, -1) / self.dim_k ** 0.5) @ v1
        head2_output = self.softmax(q2 @ k2.transpose(-2, -1) / self.dim_k ** 0.5) @ v2

        combined_heads = torch.cat((head1_output, head2_output), dim=-1)
        attention_output = self.attention_head_projection(combined_heads)

        outputs = self.norm_mh(inputs + attention_output)

        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """

        ff_out = self.ff(inputs)

        # norm
        outputs = self.norm_ff(inputs + ff_out)

        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """

        outputs = self.fl(inputs)

        return outputs


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True