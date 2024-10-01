import tiktoken
import torch
import warnings

from torch.utils.data import Dataset, DataLoader, TensorDataset

warnings.filterwarnings(action= 'ignore')

device = torch.device(torch._C._get_default_device())

torch.manual_seed(123)

#tokenizer = tiktoken.get_encoding('gpt2') # initilizer the tokenizer (BPE)
#token_ids = tokenizer.encode(raw_text)

class GPTDataSet(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []


        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride): # (0, 4889, 1) 5145 - 256 = 4889
            input_chunk = token_ids[i: i + max_length] # (0, 256) , (1, 257)
            target_chunk = token_ids[i+1: i+max_length+1] # (1, 257), (2, 257)
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]        

def create_dataset(txt, batch_size= 8, max_length= 256, stride= 128):
    
    tokenizer = tiktoken.get_encoding('gpt2')
    
    dataset = GPTDataSet(txt, tokenizer= tokenizer, max_length= max_length, stride= stride)
    #input_id_length = dataset.__len__()
    #print("Input Id Length: ", input_id_length)
    #input_ids, target_ids=  dataset.__getitem__(2)
    #return input_ids, target_ids

     
    dataloader = DataLoader(
        dataset= dataset,
        batch_size= batch_size,
        shuffle= True,
        drop_last= True
    )
    return dataloader

'''
Token and Position Embeddings
'''
class TokenAndPositionEmbedding():
    def __init__(self, max_length, context_length, vocab_size, output_dim):
        
        self.max_length = max_length
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        
        self.token_embedding_layer = torch.nn.Embedding(num_embeddings= vocab_size, embedding_dim= output_dim)
        self.pos_embedding_layer = torch.nn.Embedding(num_embeddings= context_length, embedding_dim= output_dim)
    
    def call(self, inputs): 
        token_embeddings = self.token_embedding_layer(inputs)
        pos_embeddings = self.pos_embedding_layer(torch.arange(self.max_length))
        return token_embeddings, pos_embeddings 
        
if __name__ == "__main__":
    
    filepath = "the-verdict.txt"
    batch_size = 8
    max_length = 256
    stride = 128
    vocab_size = 50257
    output_dim = 256
    context_length = 256

    with open(filepath, 'r', encoding= 'utf-8') as f:
        raw_text = f.read()

    dataloader = create_dataset(txt= raw_text, batch_size= batch_size, max_length= max_length, stride= stride)
    print("No of Batch Groups: {}".format(len(dataloader)))
    
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    # create token and Position embeddings
    objTokenAndPositionEmbedding = TokenAndPositionEmbedding(
        max_length= max_length,
        context_length= context_length,
        vocab_size= vocab_size,
        output_dim= output_dim
        )
    token_embeddings, pos_embeddings = objTokenAndPositionEmbedding.call(inputs= inputs)
    
    
    print("token_embeddings shape: {}".format(token_embeddings.shape))
    print("pos_embeddings shape: {}".format(pos_embeddings.shape))


    input_embeddings = token_embeddings + pos_embeddings
    print("input_embeddings shape: {}".format(input_embeddings.shape))
    print("\n input :{}".format(input_embeddings))
    










        