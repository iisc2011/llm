import torch.nn as nn
import torch
from tokenizer import create_dataset, TokenAndPositionEmbedding
import json

torch.manual_seed(123)
'''
To design the self attention weights for each tokens without masking
  
'''
class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, bias= False):
        super().__init__()
        self.W_query = nn.Linear(in_features= dim_in, out_features= dim_out, bias= bias) # query 
        self.W_key = nn.Linear(in_features= dim_in, out_features= dim_out, bias= bias) # key
        self.W_value = nn.Linear(in_features= dim_in, out_features= dim_out, bias= bias) # value

    def forward(self, x):
        
        keys = self.W_key(x) # calculate the keys scores 
        queries = self.W_query(x) # calculate the queries scores
        values = self.W_value(x) # calculate the values

        # get the attention scores
        attention_scores = queries @ keys.T 
        dim_k = keys.shape[-1] # get the key matrix's column dimension
        # get the attention weights (normalize using softmax)
        attention_weights = torch.softmax(attention_scores / dim_k**0.5, dim= -1) 
        
        # calculate the context vector
        context_vector = attention_weights @ values

        return context_vector

'''
Create the self attention with masking means only consider tokens prior to the current positions 
to predict the next token in sequence. This method called as Causal Attention mask

'''    
class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, bias= False):
        super().__init__()
        self.W_query = nn.Linear(in_features= dim_in, out_features= dim_out, bias= bias) # (dim_in, dim_out)
        self.W_key = nn.Linear(in_features= dim_in, out_features= dim_out, bias= bias)
        self.W_value = nn.Linear(in_features= dim_in, out_features= dim_out, bias= bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', 
                              torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape
        
        keys = self.W_key(x) # (dim_in, dim_out) * (batch, num_of_token, dim_in)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1, 2) 
        attention_scores.masked_fill_(self.mask.bool(), -torch.inf)
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1) #[batch_size,no_of_tokens,no_of_tokens]
        
        # apply the dropout
        attention_weights = self.dropout(attention_weights)

        # get the context vector
        context_vectors = attention_weights @ values

        return context_vectors
'''
Multihead attention is adding casual single attention in stacks so final attention head dimension is dim_out * num_heads but this 
process make matrix multiplication mutiple times based on num_heads and this calculation will become intensive if increase 
the num_heads , GPT 2 (117 P) having 12 num_heads and context_length = 768 (dim_out) . 

To make this procedure computationaly effective , will do multi head attenstion with weight split . Key point to implement this
to consider head_dim = dim_out / num_heads 


'''
class MultiheadAttention(nn.Module):
    '''
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads, bias= False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(dim_in, dim_out, context_length, dropout) for _ in range(num_heads)])
    
    def forward(self, x):
        attentions_heads = [head(x) for head in self.heads]
        return torch.cat(attentions_heads, dim=-1) 
    '''
    '''
    Computationaly effective ways to implement multi head attension layers 
    '''
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads, bias= False):
        super().__init__()
        self.W_query = nn.Linear(in_features= dim_in, out_features= dim_out, bias= bias)
        self.W_key = nn.Linear(in_features= dim_in, out_features= dim_out, bias= bias)
        self.W_value = nn.Linear(in_features= dim_in, out_features= dim_out, bias= bias)
        self.head_dim = dim_out // num_heads # it should be number and divisible by num_heads (dim_out // num_heads == 0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads

    def forward(self, x):

        #print('x.shape:{}'.format(x.shape))
        batch_size, num_tokens, dim_in = x.shape
        
        keys = self.W_key(x) # shape (batch_size, num_tokens, dim_out)
        queries = self.W_query(x)
        values = self.W_value(x)
        #print('values shape: {}'.format(values.shape))

        # decompose the matrix into (batch_size, num_tokens, num_heads, head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        
        # transpose 
        keys = keys.transpose(1,2) #(batch_size, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        
        #print('values shape after decompose: {}'.format(values.shape))

        attn_scores = queries @ keys.transpose(2,3)
        mask = self.mask.bool()
        attn_scores = attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim=-1) #(batch_size, num_heads, num_tokens, head_dim)
        #print('attn_weights : {}'.format(attn_weights))
        
        # reshape the attn_weight to (batch_size, num_tokens, dim_out)
        attn_weights = self.dropout(attn_weights)
        context_vector = (attn_weights @ values).transpose(1,2) 
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.dim_out)
        return context_vector

if __name__ == "__main__":
    
    # load the GPT2 model configuration
    gpt_config_file_path = 'gpt_config_124M.json'
    with open(gpt_config_file_path, 'r', encoding= 'utf-8') as gpt_config_file:
        config = json.load(gpt_config_file)
    
    #print(config['drop_rate'])


    filepath = "the-verdict.txt"
    batch_size = 8
    max_length = config['context_length'] #256
    stride = 128
    vocab_size = config['vocab_size']
    dim_out = config["emb_dim"]           #256
    context_length = config['context_length']  #256
    num_heads = config['n_heads']              #4
    dim_in = config["emb_dim"]

    dropout = 0.1 #config['dropout']


    
    with open(filepath, 'r', encoding= 'utf-8') as f:
        raw_text = f.read()
    

    dataloader = create_dataset(txt= raw_text, batch_size= batch_size, max_length= max_length, stride= stride)
    print("No of Batch Groups: {}".format(len(dataloader)))

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    # create token and Position embeddings
    tokenandpositionembedding = TokenAndPositionEmbedding(
        max_length= max_length,
        context_length= context_length,
        vocab_size= vocab_size,
        dim_out= dim_out
        )
    token_embeddings, pos_embeddings = tokenandpositionembedding.forward(inputs)
    
    
    #print("token_embeddings shape: {}".format(token_embeddings.shape))
    #print("pos_embeddings shape: {}".format(pos_embeddings.shape))


    input_embeddings = token_embeddings + pos_embeddings
    print("input shape: {}".format(input_embeddings.shape))
    context_vector = MultiheadAttention(dim_in, dim_out, context_length, dropout= dropout, num_heads= num_heads)
    context_vector = context_vector(input_embeddings)

    print('output shape: {}'.format(context_vector.shape))
    
    

    
    
    
    
        
        
