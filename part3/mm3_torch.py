import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


#get words database
words = open('names.txt', 'r').read().splitlines()

#encoder and decoder
stoi = {s : i+1 for i, s in enumerate(sorted(set(''.join(words))))}
stoi['.'] = 0
itos = {i+1: s for i, s in enumerate(sorted(set(''.join(words))))}
itos[0] = '.'
vocab_size = len(stoi)

#build dataset
block_size = 3
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        chs = list(w) + ['.'] 
        for ch in chs:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)

n1 = int(len(words)*0.8)
n2 = int(len(words)*0.9)

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


#####################################################################
class Linear():
    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros((fan_out)) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d():
    def __init__(self, dim, eps = 1e-5, momentum = 0.99):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.bngain = torch.ones(dim)
        self.bnbias = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim = True)
            xvar = x.var(0, keepdim = True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean)/torch.sqrt(xvar + self.eps)
        self.out = self.bngain * xhat + self.bnbias
        if self.training:
            with torch.no_grad():
                self.running_mean = self.running_mean * self.momentum + xmean * (1- self.momentum)
                self.running_var  = self.running_var * self.momentum + xvar * (1-self.momentum)
        return self.out
    
    def parameters(self):
        return [self.bngain, self.bnbias]


class Tanh():
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
    
g = torch.Generator().manual_seed(2147483647)
n_embd = 10
n_hidden = 100
C = torch.randn((vocab_size, n_embd), generator = g)
layers = [
    Linear(n_embd * block_size, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size), BatchNorm1d(vocab_size),
]

with torch.no_grad():
    layers[-1].bngain *= 0.1

    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

    
max_steps = 100000
batch_size = 32

lossi = []
ud = []
for i in range(max_steps):
    #batch
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator = g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass
    #print('C', C.shape)
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    #print('emb', x.shape)
    for layer in layers:
        #print('current x', x.shape)
        x = layer(x)
    
    loss = F.cross_entropy(x, Yb)
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    
    #backward pass
    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    
    lr = 0.1 if i < 100000 else 0.01
    loss.backward()
    with torch.no_grad():
        ud.append([(lr*p.grad.std() / p.data.std()).log10().item() for p in parameters])
    for p in parameters:
        p.data += -lr * p.grad
    lossi.append(loss)
    if i == 1:
        #model performace visualization
        plt.figure(figsize = (20, 4))
        legends = []
        for i, layer in enumerate(layers):
            if isinstance(layer, Tanh):
                t = layer.out
                print('layer %d (%10s): mean %+.2f, std %.2f, saturated %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
                hy, hx = torch.histogram(t, density = True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'layer {i} ({layer.__class__.__name__})')
        plt.legend(legends)
        plt.title('activation distrubution')
        plt.xlabel('activation value')
        plt.ylabel('frequency')
        plt.show()
        plt.figure(figsize = (20, 4))
        legends = []
        for i , layer in enumerate(layers):
            if isinstance(layer, Tanh):
                t = layer.out.grad
                print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
                hy, hx = torch.histogram(t, density = True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'layer {i} ({layer.__class__.__name__})')
        plt.legend(legends)
        plt.title('gradient distribution')
        plt.xlabel('gradient value')
        plt.ylabel('frequency')
        plt.show()
        plt.figure(figsize = (20, 4))
        legends = []
        for i, p in enumerate(parameters):
            t = p.grad
            if p.ndim == 2:
                print('weight %10s | mean %+f, std %e | grad: data ratio %e' % (tuple(p.shape), t.mean(), t.std(), (t.std() / p.std())))
                hy, hx = torch.histogram(t, density = True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'{i} {tuple(p.shape)}')
        plt.legend(legends)
        plt.title('weight gradient distribution')
        plt.xlabel('gradient value')
        plt.ylabel('frequency')
        plt.show()
    if i == 1000:
        plt.figure(figsize = (20, 4))
        legends = []
        for i, p in enumerate(parameters):
            if p.ndim == 2:
                plt.plot([ud[j][i] for j in range(len(ud))])
                legends.append('param %d' % i)
        plt.plot([0, len(ud)], [-3, -3], 'k')
        plt.legend(legends)
        plt.title('log10(lr*grad/weight) for each parameter')
        plt.xlabel('step')
        plt.ylabel('log10(lr*grad/weight)')
        plt.show()

#model evaluation
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'dev': (Xdev, Ydev),
        'test': (Xte, Yte)
    }[split]
    emb = C[x]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y)
    print(loss.item(), split)

split_loss('train')
split_loss('dev')



     

