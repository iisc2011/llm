import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(123)
torch.set_printoptions(sci_mode=False) # to disable scientific mode 
'''
Design the Layer Normalization layer as per
transformer architecture 
'''
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim= True)
        var = x.var(dim=-1, keepdim= True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

# Original GPT2 model was trained on GELU activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
       return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor((2 / torch.pi))) * (x + .044715 * torch.pow(x, 3))
            ))       

if __name__ == '__main__':
    '''
    Varify LayerNorm function

    batch_example = torch.randn(2,5)
    #print(batch_example)
    layer = LayerNorm(5)
    out = layer(batch_example)
    mean = out.mean(dim=-1, keepdim= True)
    var = out.var(dim=-1, keepdim= True, unbiased=False)
    print('mean: {} and var: {}'.format(mean, var))
    '''

    '''
    verify GELU Vs RELU activation function
    '''     
    x = torch.linspace(start= -3, end= 3, steps= 100)

    gelu, relu = GELU(), nn.ReLU()
    y_gelu, y_relu = gelu(x), relu(x)
    
    plt.figure(figsize= (8,3))
    
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ['GELU', 'RELU']), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f'{label} activation function')
        plt.xlabel('x')
        plt.ylabel(f'{label}(x)')
        plt.grid(True)

    plt.tight_layout()
    plt.show()    

