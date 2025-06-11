import torch
import random
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


#params:
block_size = 16
n_embd = 10
n_hidden = 200
batch_size = 64
dim = 2
max_steps = 300000

class Linear():
    def __init__(self, fan_in, fan_out, bias = True):
        self.weights = torch.randn(fan_in, fan_out) / (fan_in)**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weights
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def params(self):
        return [self.weights] + [[] if self.bias is None else self.bias]

class Tanh():
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def params(self):
        return []

class BatchNorm1d():
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.ones(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim, keepdim = True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = self.momentum * xmean + self.running_mean * (1-self.momentum)
                self.running_var = self.momentum * xvar + self.running_var * (1-self.momentum)
        return self.out
    
    def params(self):
        return [self.gamma, self.beta]

class Embedding():
    def __init__(self, num_embeddings, embedding_dim):
        self.weights = torch.randn(num_embeddings, embedding_dim)
    def __call__(self, IX):
        self.out = self.weights[IX]
        return self.out
    
    def params(self):
        return [self.weights]

class Flatten():
  def __init__(self, n):
    self.n = n
  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
    return self.out
  def params(self):
    return []

class Sequential():
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def params(self):
        return [p for layer in self.layers for p in layer.params()]

words = open('names.txt', 'r').read().splitlines()

stoi = {s:i+1 for i, s in enumerate(list(sorted((set(''.join(words))))))}
stoi['.'] = 0
itos = {i+1:s for i, s in enumerate(list(sorted((set(''.join(words))))))}
itos[0] = '.'
vocab_size = len(stoi)

random.seed(42)
random.shuffle(words)

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")
    return X, Y

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


model = Sequential([
    Embedding(vocab_size, n_embd),Flatten(dim),
    Linear(n_embd * dim, n_hidden), BatchNorm1d(n_hidden),Tanh(), Flatten(dim),
    Linear(n_hidden*dim, n_hidden), BatchNorm1d(n_hidden),Tanh(),Flatten(dim),
    Linear(n_hidden*dim, n_hidden), BatchNorm1d(n_hidden),Tanh(),Flatten(dim),
    Linear(n_hidden*dim, n_hidden), BatchNorm1d(n_hidden),Tanh(),
    Linear(n_hidden, vocab_size)
])

for layer in model.layers:
    print(layer.__class__.__name__,":", tuple(layer.out.shape) if hasattr(layer, 'out') else None)

params = model.params()
print(sum(p.nelement() for p in params), 'parameters')

with torch.no_grad():
    model.layers[-1].weights *= 0.1

for p in params:
    p.requires_grad = True


for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
    Xb, Yb = Xtr[ix], Ytr[ix]

    if i == 1:
        print(f'vocab_size: {vocab_size} | block_size: {block_size} | n_embd: {n_embd} | n_hidden: {n_hidden} | batch_size: {batch_size}')
        print(f'Xtr: {Xtr.shape} | ix {ix.shape} | Xb: {Xb.shape}')

    logits = model(Xb)
    
    loss = F.cross_entropy(logits, Yb)
    if i % 10000 == 0:
        print('loss: ',f'step {i} / {max_steps}' ,loss.item())

    for p in params:
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 100000 else 0.01 if i < 200000 else 0.005

    for p in params:
        p.data += -lr * p.grad

for layer in model.layers:
    layer.training = False  

@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte)
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())
    print('avg model confidence', f'{math.exp(-loss.item())*100:.2f}%')

split_loss('train')
split_loss('val')

for i in range(20):
    context = [0] * block_size
    out = []
    while True:
        x = torch.tensor(context).unsqueeze(0)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples = 1).item()
        out.append(itos[ix])
        context = context[1:] + [ix]
        if ix == 0:
          break
    print(''.join(out[:-1]))