import torch
import torch.nn.functional as F


words = open('names.txt', 'r').read().splitlines()


#create stoi and itos: a list with all ordered letters inside words dataset
#stoi: letter: ordered number
#itos: ordered number: letter
#. is or beginning or end of word: it has 0 as assigned value
stoi = {s:i+1 for i,s in enumerate(sorted(list(set(''.join(words)))))}
stoi['.'] = 0
itos = {i+1:s for i,s in enumerate(sorted(list(set(''.join(words)))))}
itos[0] = '.'

#initialize inputs (xs) and expected result (ys)
xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)


#INITIALIZE MODEL:

#create a generator g for random number and initialize with following number: 2147483647
g = torch.Generator().manual_seed(2147483647)
#create a 27, 27 torch matrix with random weights (using generator g)
#its a 1 Layer with 27 neurons, each one of them gets 27 inputs
#initialize it with grad
W = torch.randn((27,27), generator = g, requires_grad=True)


#FORWARD PASS:

#encode x inputs using one hot enconding, matrix columns = 27, matrix rows depend on xs length
#dtype has to be float not int, otherwise matrix multiplication would not be possible with W
xenc = F.one_hot(xs, num_classes = 27).float()
#multiply xenc * W to get probability distribution for each lettter for each neuron
logits = xenc @ W
#exponentiate logits since some of values can have negative values: not possible to get probability distribuition with negative valeus
counts = logits.exp()
#get probs by dividing each row element by row sum. to get probability per row
probs = counts / counts.sum(1, keepdim = True)
#calculate loss function:
#evaluate log function for each row in probs with ys index (evaluate probability that with x input you get the expected value (y))
#calculate negative log fuunction and calculate mean
loss = -probs[torch.arange(5), ys].log().mean()
print(loss)

#BACKWARD PASS:

#intialize grad as None in order to not sum grads for each loop:
W.grad = None
#calculate gradient using backward function:
loss.backward()

#recalculate W weights based on gradient:
#to minimize loss directions should be opposite of the gradient:
W.data += -0.1 * W.grad






