"""
5-Embedding Model Implementation
Combines BERT, ELMo, GloVe, POS, Lexicon, and Character embeddings for sentiment analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import List
from transformers import BertModel


class ModernELMo:
    """
    Modern ELMo implementation using TensorFlow Hub
    """
    def __init__(self):
        self.elmo_url = "https://tfhub.dev/google/elmo/3"
        print("Loading ELMo model...")
        self.elmo_model = hub.load(self.elmo_url)
        print("ELMo model loaded successfully!")

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def _get_embeddings_tf(self, sentences):
        return self.elmo_model.signatures['default'](sentences)['elmo']

    def get_contextualized_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Get contextualized embeddings for a list of sentences
        
        Args:
            sentences: List of input sentences
            
        Returns:
            numpy array of ELMo embeddings
        """
        sentences_tensor = tf.constant([str(s) for s in sentences])
        return self._get_embeddings_tf(sentences_tensor).numpy()


class ELMoEmbeddingLayer(nn.Module):
    """
    PyTorch wrapper for ELMo embeddings
    """
    def __init__(self):
        super().__init__()
        # Initialize self.elmo in first forward call for DataLoader worker compatibility
        self.elmo = None

    def forward(self, sentences: list, device) -> torch.Tensor:
        """
        Forward pass to get ELMo embeddings
        
        Args:
            sentences: List of input sentences
            device: Target device (CPU/GPU)
            
        Returns:
            torch.Tensor of ELMo embeddings
        """
        if self.elmo is None:
            self.elmo = ModernELMo()
        embeddings = self.elmo.get_contextualized_embeddings(sentences)
        return torch.tensor(embeddings, dtype=torch.float32).to(device)


class BertBranch(nn.Module):
    """
    BERT branch of the model
    """
    def __init__(self, bert_model, dropout):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Forward pass through BERT
        
        Args:
            input_ids: BERT input token IDs
            attention_mask: BERT attention mask
            token_type_ids: BERT token type IDs
            
        Returns:
            Dropout-applied BERT hidden states
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.dropout(outputs.last_hidden_state)


class GloveBranch(nn.Module):
    """
    GloVe embeddings branch
    """
    def __init__(self, vocab_size, embed_dim, dropout, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.requires_grad = not freeze
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        """
        Forward pass through GloVe embeddings
        
        Args:
            token_ids: Token IDs for GloVe lookup
            
        Returns:
            GloVe embeddings with dropout applied
        """
        return self.dropout(self.embedding(token_ids))


class PosBranch(nn.Module):
    """
    Part-of-Speech embeddings branch
    """
    def __init__(self, pos_vocab_size, pos_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(pos_vocab_size, pos_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pos_ids):
        """
        Forward pass through POS embeddings
        
        Args:
            pos_ids: POS tag IDs
            
        Returns:
            POS embeddings with dropout applied
        """
        return self.dropout(self.embedding(pos_ids))


class LexiconBranch(nn.Module):
    """
    Lexicon embeddings branch
    """
    def __init__(self, vocab_size, lex_dim, dropout, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, lex_dim, padding_idx=0)
        self.embedding.weight.requires_grad = not freeze
        self.dropout = nn.Dropout(dropout)

    def forward(self, lex_ids):
        """
        Forward pass through lexicon embeddings
        
        Args:
            lex_ids: Lexicon IDs
            
        Returns:
            Lexicon embeddings with dropout applied
        """
        return self.dropout(self.embedding(lex_ids))


class CharBiLSTMBranch(nn.Module):
    """
    Character-level BiLSTM embeddings branch
    """
    def __init__(self, char_vocab_size, char_dim, char_hidden, dropout):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
        self.bilstm = nn.LSTM(char_dim, char_hidden, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = 2 * char_hidden

    def forward(self, char_ids):
        """
        Forward pass through character BiLSTM
        
        Args:
            char_ids: Character IDs tensor [B, T, C]
            
        Returns:
            Character-level embeddings with dropout applied
        """
        B, T, C = char_ids.shape
        char_ids_flat = char_ids.view(-1, C)
        embeddings = self.embedding(char_ids_flat)
        outputs, _ = self.bilstm(embeddings)
        outputs = outputs[:, -1, :]  # Take last output
        outputs = outputs.view(B, T, self.out_dim)
        return self.dropout(outputs)


class GaussianNoise(nn.Module):
    """
    Gaussian noise layer for regularization
    """
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, x):
        if self.training:
            return x + self.sigma * torch.randn_like(x)
        return x


class AdditiveAttention(nn.Module):
    """
    Additive attention mechanism
    """
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, attn_dim)
        self.linear2 = nn.Linear(attn_dim, 1, bias=False)
        
    def forward(self, H, mask):
        """
        Apply additive attention
        
        Args:
            H: Hidden states
            mask: Attention mask
            
        Returns:
            Tuple of (context vector, attention weights)
        """
        H_proj = torch.tanh(self.linear1(H))
        attn_scores = self.linear2(H_proj).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(attn_scores, dim=1)
        context = torch.sum(alpha.unsqueeze(-1) * H, dim=1)
        return context, alpha


class FiveEmbeddingModel(nn.Module):
    """
    5-Embedding Model combining BERT, ELMo, GloVe, POS, Lexicon, and Character embeddings
    """
    def __init__(self, num_classes):
        super().__init__()
        # Initialize all embedding branches
        self.bert_branch = BertBranch("bert-base-uncased", 0.25)
        self.elmo_layer = ELMoEmbeddingLayer()
        self.glove_branch = GloveBranch(30000, 300, 0.25)
        self.pos_branch = PosBranch(64, 50, 0.25)
        self.lex_branch = LexiconBranch(30000, 6, 0.25)
        self.char_branch = CharBiLSTMBranch(128, 50, 25, 0.25)

        # Calculate total dimension: BERT(768) + ELMo(1024) + GloVe(300) + POS(50) + Lexicon(6) + Char(50)
        total_dim = 768 + 1024 + 300 + 50 + 6 + 50
        self.encoder = nn.LSTM(
            input_size=total_dim, 
            hidden_size=256, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.attn = AdditiveAttention(hidden_dim=2*256, attn_dim=128)
        self.fc = nn.Sequential(
            nn.Linear(2*256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
        self.noise = GaussianNoise(0.3)

    def _align_tensor(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Align tensor length to target length by truncating or padding
        
        Args:
            tensor: Input tensor
            target_len: Target sequence length
            
        Returns:
            Aligned tensor
        """
        current_len = tensor.size(1)
        if current_len == target_len:
            return tensor
        elif current_len > target_len:
            return tensor[:, :target_len, :]
        else:
            padding_size = target_len - current_len
            paddings = (0, 0, 0, padding_size)
            return F.pad(tensor, paddings, "constant", 0)

    def forward(self, batch):
        """
        Forward pass through the 5-Embedding model
        
        Args:
            batch: Dictionary containing input data with keys:
                - input_ids: BERT input token IDs
                - attention_mask: BERT attention mask
                - token_type_ids: BERT token type IDs
                - sentences: List of input sentences for ELMo
                - token_ids: Token IDs for GloVe
                - pos_ids: POS tag IDs
                - lex_ids: Lexicon IDs
                - char_ids: Character IDs
                
        Returns:
            Dictionary containing:
                - logits: Classification logits
                - attn: Attention weights
        """
        # BERT branch
        bert_rep = self.bert_branch(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids']
        )
        target_len = bert_rep.size(1)
        device = bert_rep.device

        # ELMo branch
        elmo_rep = self.elmo_layer(batch['sentences'], device)
        elmo_rep = self._align_tensor(elmo_rep, target_len)

        # GloVe branch
        glove_rep = self.glove_branch(batch['token_ids'])
        glove_rep = self._align_tensor(glove_rep, target_len)

        # POS branch
        pos_rep = self.pos_branch(batch['pos_ids'])
        pos_rep = self._align_tensor(pos_rep, target_len)

        # Lexicon branch
        lex_rep = self.lex_branch(batch['lex_ids'])
        lex_rep = self._align_tensor(lex_rep, target_len)

        # Character branch
        char_rep = self.char_branch(batch['char_ids'])
        char_rep = self._align_tensor(char_rep, target_len)

        # Concatenate all representations
        reps = [bert_rep, elmo_rep, glove_rep, pos_rep, lex_rep, char_rep]
        X = torch.cat(reps, dim=-1)
        X = self.noise(X)

        # LSTM encoder
        H, _ = self.encoder(X)
        
        # Attention mechanism
        attention_mask = self._align_tensor(batch['attention_mask'].unsqueeze(-1), target_len).squeeze(-1)
        context, alpha = self.attn(H, mask=attention_mask)
        
        # Classification head
        logits = self.fc(context)

        return {"logits": logits, "attn": alpha}
