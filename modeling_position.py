import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import warnings

from typing import List, Tuple


# pylint: disable=no-member

# try:
#     from .cuda_gelu import GELU
#     gelu = GELU()
# except:
#     warnings.warn(
#         "Try to install Cupy to enable CUDA gelu activation function!")

def gelu(x):
    """Implementation of the gelu activation function.
    """
    return x * 0.5 * (1.0 + torch.erf(x * 0.707106781186547461715))


try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except:
    warnings.warn("Try to install apex to improve your performance!")


    class LayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class TransformerSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / 12)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        self.coefficient = torch.nn.Parameter(torch.FloatTensor([[1,1,0,0,0,0,0,0,0,0,0,1],
                                                                 [1,1,1,1,0,0,0,0,0,0,0,1],
                                                                 [1,1,1,1,1,1,0,0,0,0,0,1],
                                                                 [1,1,1,1,1,1,1,1,0,0,0,1],
                                                                 [1,1,1,1,1,1,1,1,1,1,0,1]]))

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[
                      :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, layer_past, mask, position_mask=None, encoder_turn=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # FIX: potential error her
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)

        present = torch.stack((key_layer, value_layer), dim=0)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / 8.0
        ################自己加入
        if position_mask is not None and encoder_turn is not None:
            # encoder_turn = encoder_turn - torch.ones(encoder_turn.shape[0])
            encoder_turn = torch.div(encoder_turn-torch.ones(encoder_turn.shape[0],device="cuda:0"), 2)
            encoder_turn = torch.tensor(encoder_turn.unsqueeze(-1).cpu().numpy(), device="cuda:0", dtype=torch.int64)
            encoder_turn_mask = torch.zeros(encoder_turn.shape[0], self.coefficient.shape[0], device="cuda:0").scatter_(1, encoder_turn, 1)
            coefficient = torch.matmul(encoder_turn_mask, self.coefficient).unsqueeze(-1)
            position_score = torch.matmul(position_mask, coefficient).squeeze(-1)
            position_score = position_score.repeat(1, attention_scores.shape[1]*attention_scores.shape[2])
            position_score = position_score.reshape(attention_scores.shape)
            attention_scores = torch.mul(attention_scores, position_score)
        nd, ns = attention_scores.size(-2), attention_scores.size(-1)

        mask = mask[:, :, ns - nd:ns, :ns]

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores * mask - 1e10 * (1 - mask)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # return two tensors
        return context_layer, present


class TransformerSelfOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = TransformerSelfAttention()
        self.output = TransformerSelfOutput()

    def forward(self, input_tensor, layer_past, mask, position_mask=None, encoder_turn=None):
        self_output, present = self.self(input_tensor, layer_past, mask, position_mask, encoder_turn)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, present


class TransformerIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 3072)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TransformerOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TransformerAttention()
        self.intermediate = TransformerIntermediate()
        self.output = TransformerOutput()

    def forward(self, hidden_states, layer_past, mask, position_mask=None, encoder_turn=None):
        attention_output, present = self.attention(
            hidden_states, layer_past, mask, position_mask, encoder_turn)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, present


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([TransformerLayer() for _ in range(12)])

    def forward(self, hidden_states, mask, past: List, position_mask=None, encoder_turn=None) -> Tuple[torch.Tensor, List]:
        presents = []

        for layer_block, layer_past in zip(self.layer, past):
            hidden_states, present = layer_block(
                hidden_states, layer_past, mask, position_mask, encoder_turn)
            presents.append(present)

        return hidden_states, presents


class TransformerDecoderEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(21128, 768)
        self.position_embeddings = nn.Embedding(512, 768)
        self.token_type_embeddings = nn.Embedding(2, 768)

        self.LayerNorm = LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, past_length, token_type_ids=None):
        position_ids = torch.arange(
            past_length, input_ids.shape[-1] + past_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = TransformerDecoderEmbeddings()
        self.encoder = TransformerModel()

    def forward(self, input_ids, mask, past=None, past_length=None, encoder_position_id=None):
        """
        mask: [batch_size, seq_length] is attention mask
        """
        # Fast way to compute lower triangle attention mask
        mask = mask.to(dtype=torch.uint8)
        mask = mask.view(input_ids.shape[0], 1, 1, -1).expand(
            input_ids.shape[0], 12, mask.shape[1], mask.shape[1])
        # mask = (mask + mask.permute(0, 1, 3, 2)) / 2
        mask = torch.floor_divide(mask + mask.permute(0, 1, 3, 2), 2)

        # fp16 compatibility
        mask = mask.to(dtype=next(self.parameters()).dtype)
        # lower triangle matrix
        mask = torch.tril(mask)
        ##############################自己加入的position_mask
        encoder_turn = encoder_position_id.max(-1).values
        encoder_position_id = torch.unsqueeze(encoder_position_id, -1)
        encoder_position_mask = torch.zeros(encoder_position_id.shape[0], encoder_position_id.shape[1], 12, device="cuda:0").scatter_(2, encoder_position_id, 1)
        decoder_position_mask = torch.tensor([[[0]*11+[1]]*(mask.shape[-1]-encoder_position_id.shape[1])]*input_ids.shape[0], device="cuda:0", dtype=torch.float)
        position_mask = torch.cat([encoder_position_mask, decoder_position_mask],1)

        # past length calculation and dealing with past
        if past is None:
            past_length = 0
            past = [None] * 12
        else:
            if past_length is None:
                past_length = past[0][0].size(-2)
            else:
                past_length = past_length

        # calculate embedding output
        embedding_output = self.embeddings(input_ids, past_length)

        # Transformer layer
        last_layer_output, presents = self.encoder(embedding_output,
                                                   mask,
                                                   past,
                                                   position_mask,
                                                   encoder_turn)

        return last_layer_output, presents


class TransformerDecoderLM(nn.Module):
    """Transformer Decoder Language Model
    This module computes the logits output and have a embedding projection matrix.
    """

    def __init__(self):
        super().__init__()
        self.transformer = TransformerDecoder()
        self.projection = nn.Linear(768, 21128, bias=False)

        self.set_tied()

    def set_tied(self):
        """Although weights are set tied explicitly here, when loading new weights,
        call it again.
        """
        self.projection.weight = self.transformer.embeddings.word_embeddings.weight

    def forward(self, input_ids, mask, past=None, past_length=None, encoder_position_id=None):
        hidden_states, presents = self.transformer(
            input_ids, mask, past, past_length, encoder_position_id)
        lm_logits = self.projection(hidden_states)
        return lm_logits, presents


class TransformerEncoderEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(21128, 768)
        self.position_embeddings = nn.Embedding(512, 768)
        self.token_type_embeddings = nn.Embedding(2, 768)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, past_length, token_type_ids=None):
        position_ids = torch.arange(
            past_length, input_ids.shape[-1] + past_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Similarity(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = torch.mean(hidden, -2)  # (faq_len,768)
        self.linear = nn.Linear(768*2, 1)

    def forward(self, q_hidden, recall_mask):
        self.hidden = torch.unsqueeze(self.hidden, 0)
        self.hidden = self.hidden.expand(q_hidden.size()[0], self.hidden.size()[1], self.hidden.size()[2])  # (batch, faq_len,768)
        q_hidden = torch.mean(q_hidden, -2)  # (batch,768)
        q_hidden = torch.unsqueeze(q_hidden, 1)
        q_hidden = q_hidden.expand(q_hidden.size()[0], self.hidden.size()[1], q_hidden.size()[2]) # (batch, faq_len,768)

        multiply = torch.mul(self.hidden, q_hidden)
        subtract = torch.abs(self.hidden - q_hidden)
        output_layer = torch.cat((multiply, subtract), -1)  # (batch,faq_len, 768*2)
        score = self.linear(output_layer)
        recall_mask = recall_mask.unsqueeze(-1)
        score = score.mul(recall_mask)

        return score


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = TransformerEncoderEmbeddings()
        self.encoder = TransformerModel()

    def forward(self, input_ids, mask, token_type_ids=None, past=None):
        """
        mask: [batch_size, seq_length] is attention mask
        """
        # Fast way to compute lower triangle attention mask
        mask = mask.to(dtype=torch.uint8)
        mask = mask.view(input_ids.shape[0], 1, 1, -1).expand(
            input_ids.shape[0], 12, mask.shape[1], mask.shape[1])
        # mask = (mask + mask.permute(0, 1, 3, 2)) / 2
        mask = torch.floor_divide(mask + mask.permute(0, 1, 3, 2), 2)  # 加起来除以2, uint8 得到0/1
        # fp16 compatibility
        mask = mask.to(dtype=next(self.parameters()).dtype)

        # past length calculation and dealing with past
        if past is None:
            past_length = 0
            past = [None] * 12
        else:
            past_length = past[0][0].size(-2)

        # calculate embedding output
        embedding_output = self.embeddings(
            input_ids, past_length, token_type_ids)

        # Transformer layer
        last_layer_output, presents = self.encoder(embedding_output,
                                                   mask,
                                                   past)

        return last_layer_output, presents