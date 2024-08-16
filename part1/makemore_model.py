import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()

stoi = {s:i+1 for i,s in enumerate(sorted(list(set(''.join(words)))))}
stoi['.'] = 0
itos = {i+1:s for i,s in enumerate(sorted(list(set(''.join(words)))))}
itos[0] = '.'


xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f"num of examples: {num}")

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator = g, requires_grad= True)

def simple_makemore(num_of_examples, num_iterations=100, learning_rate=1, regularization_index=0.01):
    print(num_of_examples)
    for k in range(num_iterations):

        #forward pass
        xenc = F.one_hot(xs, num_classes= 27).float() #input to the network: one hot encoding
        logits = xenc @ W # predict log counts (positive + negatives)
        counts = logits.exp() #counts (only positive)
        probs = counts / counts.sum(1, keepdims=True) #probabilities for next character
        loss = -probs[torch.arange(num_of_examples), ys].log().mean() + regularization_index*(W**2).mean() #last sum tries to add values to 0s values - it regularize model in order to not have 0s values
        print(loss.item())
        
        #backward pass
        W.grad = None
        loss.backward()

        #update weights
        W.data += -learning_rate * W.grad
    return W

W = simple_makemore(num_of_examples= num, num_iterations=500, learning_rate=50)


#MODEL TESTING - it should generate new words similar to the ones in the dataset
#generator
g = torch.Generator().manual_seed(2147483647)


for i in range(50):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes= 27).float() #input to the network: one hot encoding
        logits = xenc @ W # predict log counts (positive + negatives)
        counts = logits.exp() #counts (only positive)
        p = counts / counts.sum(1, keepdims=True) #probabilities for next character
        ix = torch.multinomial(p, num_samples=1, replacement = True, generator = g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))