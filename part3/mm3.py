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

#neural network initialization
n_embd = 10
n_hidden = 100
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator = g)
W1 = torch.randn((block_size * n_embd, n_hidden), generator = g) * 5/3 / (block_size * n_embd) ** 0.5
b1 = torch.randn((n_hidden), generator = g) * 0
W2 = torch.randn((n_hidden, vocab_size), generator = g) * 0.01
b2 = torch.randn((vocab_size), generator = g) * 0

bngain = torch.ones(1, n_hidden)
bnbias = torch.zeros(1, n_hidden)
bnmean_running = torch.zeros(1, n_hidden)
bnstd_running = torch.ones(1, n_hidden)

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

max_steps = 200001
batch_size = 32

lossi = []
for i in range(max_steps):
    #batch
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator = g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    #Linear Layer
    #-----------------------------------------------------------------
    hpreact = embcat @ W1 + b1

    bnmeani = hpreact.mean(0, keepdim = True)
    bnstdi = hpreact.std(0, keepdim=True)

    hpreact = bngain * (hpreact - bnmeani)/ bnstdi + bnbias

    with torch.no_grad():
        bnmean_running = 0.99 * bnmean_running + 0.01 * bnmeani
        bnstd_running = 0.99 * bnstd_running + 0.01 * bnstdi
    #Non-Linear Layer
    #-----------------------------------------------------------------
    h = torch.tanh(hpreact)
    
    #Linear Layer
    #-----------------------------------------------------------------
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    if i % 10000 == 0:
       print(f'{i} / {max_steps-1}: {loss.item()}')

    #backward pass
    for p in parameters:
        p.grad = None
    
    lr = 0.1 if i < 100000 else 0.01
    loss.backward()
    for p in parameters:
        p.data += -lr * p.grad
    lossi.append(loss)

#model evaluation
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'dev': (Xdev, Ydev),
        'test': (Xte, Yte)
    }[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(loss.item(), split)

split_loss('train')
split_loss('dev')

#plotting hpreact
plt.hist(hpreact.view(-1).tolist(), 50)
plt.title('hpreact weights distribution')
plt.xlabel('value')
plt.ylabel('frequence')
plt.show()

#plotting h
plt.hist(h.view(-1).tolist(), 50)
plt.title('h weights distribution')
plt.xlabel('value')
plt.ylabel('frequence')
plt.show()

#plotting h activations
plt.figure(figsize=(10, 4))
plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')
plt.title('h activations')
plt.ylabel('input')
plt.xlabel('neuron')
plt.show()

#sample from the model

for i in range(20):
    word = []
    context  = [0] * block_size
    while True:
        emb = C[context]
        embcat = emb.view(1, -1)
        hpreact = embcat @ W1 + b1
        hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim = 1)
        ix = torch.multinomial(probs, num_samples =1, generator = g).item()
        word.append(itos[ix])
        context = context[1:] + [ix]
        if ix == 0:
            break
    print(''.join(word))



     

