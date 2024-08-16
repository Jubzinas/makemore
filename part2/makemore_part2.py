import torch
import numpy
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt


#read in all the words
words = open('names.txt', 'r').read().splitlines()

#build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

#PARAMETER 1:
block_size = 4
#create function to build the model with a specific block size:
def build_dataset(words, block_size = 3):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in list(w) + ['.']:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

#create training set (80%), dev set (10%), test set (10%)
random.seed(42)
random.shuffle(words)
n1 = int(0.8* len(words))
n2 = int(0.9* len(words))

Xtr, Ytr = build_dataset(words[:n1], block_size)
Xdev, Ydev = build_dataset(words[n1:n2], block_size)
Xte, Yte = build_dataset(words[n2:], block_size)

#initialize C, Weights, Biases
def initialize_model(block_size = 4, c_index = 6, n_neurons= 200):
    n_layer_inputs = block_size * c_index
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((27, c_index), generator = g)
    W1 = torch.randn((n_layer_inputs, n_neurons), generator = g)
    b1 = torch.randn((n_neurons), generator = g)
    W2 = torch.randn((n_neurons, 27), generator =g)
    b2 = torch.randn((27), generator = g)
    parameters = [C, W1, b1, W2, b2]
    n_parameters = sum(p.nelement() for p in parameters)
    print("number of parameters",n_parameters)
    for p in parameters:
        p.requires_grad = True
    return C, W1, b1, W2, b2, parameters, n_layer_inputs
C, W1, b1, W2, b2, parameters, n_layer_inputs = initialize_model()

#forward pass:
def f_b_pass(n_iterations = 100, lr = 0.01, batch_size = 50):
    for i in range(n_iterations):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        emb = C[Xtr[ix]]
        emb_m = emb.view(-1, n_layer_inputs)
        h = torch.tanh(emb_m @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])
        #backward pass:
        for p in parameters:
            p.grad = None
        loss.backward()
        lr = lr if i < 30000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad
    print(loss.item())
f_b_pass(100000, 0.1, 50)

#evaluate on train set
emb = C[Xtr]
h = torch.tanh(emb.view(-1, n_layer_inputs) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print('train set loss', loss.item())

#evaluate on dev set
emb = C[Xdev]
h = torch.tanh(emb.view(-1, n_layer_inputs) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print('dev set loss', loss.item())

#sampling from the model
g = torch.Generator().manual_seed(2147483647 + 101)
block_size = 4
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor(context)]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim = 1)
        ix = torch.multinomial(probs, num_samples = 1, generator = g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))