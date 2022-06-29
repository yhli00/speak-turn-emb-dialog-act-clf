import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader
import logging
import math
import sys

logger = logging.getLogger(__name__)


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.parse_query = nn.Linear(config.hidden_size, self.attention_head_size)
        self.parse_key = nn.Linear(config.hidden_size, self.attention_head_size)
        self.parse_value = nn.Linear(config.hidden_size, self.attention_head_size)

        self.mlp = nn.Linear(self.all_head_size + self.attention_head_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_parse(self, x):
        new_x_shape = x.size()[:-1] + (1, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, span_mask=None):
        # bsz, seq_len, dim
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # bsz, seq_len, head_size
        # mixed_parse_query_layer = self.parse_query(hidden_states)
        # mixed_parse_key_layer = self.parse_key(hidden_states)
        mixed_parse_value_layer = self.parse_value(hidden_states)

        # bsz, 1, seq_len, head_size
        # parse_query_layer = self.transpose_for_scores_parse(mixed_parse_query_layer)
        # parse_key_layer = self.transpose_for_scores_parse(mixed_parse_key_layer)
        parse_value_layer = self.transpose_for_scores_parse(mixed_parse_value_layer)
        if span_mask is not None:
            parse_context_layer = torch.matmul(span_mask, parse_value_layer)
            parse_context_layer = parse_context_layer.permute(0, 2, 1, 3).contiguous()
        # bsz, 1, seq_len, seq_len
        # parse_score = torch.matmul(parse_query_layer, parse_key_layer.transpose(-1,-2))

        # bsz, num_head, seq_len, head_size
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # bsz, num_head, seq_len, seq_len
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # if span_mask is not None:
        #     print("att_score", attention_scores.shape)
        #     print("att_mask", attention_mask.shape)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        if span_mask is not None:
            context_layer = torch.cat([context_layer, parse_context_layer], dim=-2)
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size + self.attention_head_size,)
            # bsz, seq_len, (num_head+1)*head_size
            context_layer = context_layer.view(*new_context_layer_shape)
            context_layer = self.mlp(context_layer)
        else:
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, span_mask=None):
        self_output = self.self(input_tensor, attention_mask, span_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, str)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, span_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, span_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class BertRNN(nn.Module):
    def __init__(self, nlayer, nclass, dropout=0.5, nfinetune=0, speaker_info='none', topic_info='none', emb_batch=0):
        super(BertRNN, self).__init__()

        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained('roberta-base')
        nhid = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False
        n_layers = 12
        logger.info(f'{nfinetune}')
        if nfinetune > 0:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers-1, n_layers-1-nfinetune, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        # classifying act tag
        self.encoder = nn.GRU(nhid, nhid//2, num_layers=nlayer, dropout=dropout, bidirectional=True)
        # self.attention = BertLayer(self.bert.config)
        # self.attentions = nn.ModuleList([BertLayer(self.bert.config) for _ in range(3)])
        self.fc = nn.Linear(nhid, nclass)

        # making use of speaker info
        self.speaker_emb = nn.Embedding(3, nhid)

        # making use of topic info
        self.topic_emb = nn.Embedding(100, nhid)

        self.dropout = nn.Dropout(p=dropout)
        self.nclass = nclass
        self.speaker_info = speaker_info
        self.topic_info = topic_info
        self.emb_batch = emb_batch

    def forward(self, input_ids, attention_mask, chunk_lens, speaker_ids, topic_labels, chunk_attention_mask):
        '''
        chunk_attention_mask: [B, chunk_size]  用来计算self-attention
        '''
        chunk_lens = chunk_lens.to('cpu')
        batch_size, chunk_size, seq_len = input_ids.shape
        speaker_ids = speaker_ids.reshape(-1)   # (batch_size, chunk_size) --> (batch_size*chunk_size)
        # chunk_lens = chunk_lens.reshape(-1)   # (batch_size, chunk_size) --> (batch_size*chunk_size)
        topic_labels = topic_labels.reshape(-1)   # (batch_size, chunk_size) --> (batch_size*chunk_size)

        input_ids = input_ids.reshape(-1, seq_len)  # [B, chunk_size, L]->[B*chunk_size, L]
        attention_mask = attention_mask.reshape(-1, seq_len)  # [B, chunk_size, L]->[B*chunk_size, L]
        if self.emb_batch == 0:
            # embeddings = self.bert(input_ids, attention_mask=attention_mask,
            #                        output_hidden_states=True)[0][:, 0]  # (bs*chunk_size, emb_dim)
            embeddings = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B*chunk_size, E]
        else:
            embeddings_ = []
            dataset2 = TensorDataset(input_ids, attention_mask)
            loader = DataLoader(dataset2, batch_size=self.emb_batch)
            for _, batch in enumerate(loader):
                # embeddings = self.bert(batch[0], attention_mask=batch[1], output_hidden_states=True)[0][:, 0]
                embeddings = self.bert(batch[0], attention_mask=batch[1]).last_hidden_state[:, 0, :]  # [emb_batch, E]
                embeddings_.append(embeddings)
            embeddings = torch.cat(embeddings_, dim=0)

        # 在bert之后加了一个dropout
        embeddings = self.dropout(embeddings)

        nhid = embeddings.shape[-1]

        if self.speaker_info == 'emb_cls':
            speaker_embeddings = self.speaker_emb(speaker_ids)  # (bs*chunk_size, emb_dim)
            embeddings = embeddings + speaker_embeddings    # (bs*chunk_size, emb_dim)
        if self.topic_info == 'emb_cls':
            topic_embeddings = self.topic_emb(topic_labels)     # (bs*chunk_size, emb_dim)
            embeddings = embeddings + topic_embeddings  # (bs*chunk_size, emb_dim)

        # reshape BERT embeddings to fit into RNN
        embeddings = embeddings.reshape(-1, chunk_size, nhid)  # (bs, chunk_size, emd_dim)

        # chunk_attention_mask_extend = chunk_attention_mask.unsqueeze(-2).repeat(1, chunk_size, 1)
        # chunk_attention_mask_extend = chunk_attention_mask_extend.unsqueeze(1)  # [B, 1, chunk_size, chunk_size]

        # outputs = embeddings
        # for attention in self.attentions:
        #     outputs = attention(outputs, chunk_attention_mask_extend)  # [B, chunk_size, emb_dim]
        # outputs = self.attention(embeddings, chunk_attention_mask_extend)  # [B, chunk_size, emb_dim]

        embeddings = embeddings.permute(1, 0, 2)  # (chunk_size, bs, emb_dim)
        assert embeddings.shape[0] == chunk_size

        # sequence modeling of act tags using RNN
        embeddings = pack_padded_sequence(embeddings, chunk_lens, enforce_sorted=False)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        outputs, _ = pad_packed_sequence(outputs)  # (max(chunk_lens), bs, emb_dim)
        if outputs.shape[0] < chunk_size:
            outputs_padding = torch.zeros(chunk_size - outputs.shape[0], batch_size, nhid, device=outputs.device)
            outputs = torch.cat([outputs, outputs_padding], dim=0)  # (chunk_len, bs, emb_dim)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)  # (chunk_size, bs, ncalss)
        # outputs = outputs.permute(1, 0, 2)  # (bs, chunk_size, nclass)
        outputs = outputs.reshape(-1, self.nclass)  # (bs*chunk_size, nclass)

        return outputs