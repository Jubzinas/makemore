import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import math

n_batch = 64
n_tok = 150
n_emb = 48
learning_rate = 1e-2
max_iters = 10000
eval_iters = 300
eval_interval = 1000
dropout = 0.2
n_heads = 6
n_layers = 5
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
n_vocab = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[n] for n in l])

data = torch.tensor(encode(text), dtype= torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#randint test

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - n_tok , (n_batch,))
    x = torch.stack([data[i:i+n_tok] for i in ix])
    y = torch.stack([data[i+1:i+n_tok+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'dev']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_s):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_s) for _ in range(n_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x)for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Head(nn.Module):
    def __init__(self, head_s):
        super().__init__()
        self.query = nn.Linear(n_emb, head_s)
        self.key = nn.Linear(n_emb, head_s)
        self.value = nn.Linear(n_emb, head_s)
        self.register_buffer('tril', torch.tril(torch.ones(n_tok, n_tok)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        self.q = self.query(x)
        self.k = self.key(x)
        self.v = self.value(x)
        wei = self.q @ self.v.transpose(-2,-1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ self.v
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
           nn.Linear(n_emb, 4*n_emb),
           nn.ReLU(),
           nn.Linear(4*n_emb, n_emb),
           nn.Dropout(dropout) 
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_emb, n_heads):
        super().__init__()
        head_s = n_emb // n_heads
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        self.sa = MultiHeadAttention(n_heads, head_s)
        self.ffwd = FeedForward(n_emb)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, n_vocab):
        super().__init__()
        self.token_embedding_table = nn.Embedding(n_vocab, n_emb)
        self.pos_embedding_table = nn.Embedding(n_tok, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, n_vocab, bias=False)
    
    def forward(self, idx, targets= None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            B, T , C = logits.shape
            logits = logits.view(B * T, C)
            B, T = targets.shape
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        print('idx.shape', idx.shape)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -n_tok:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

model = BigramLanguageModel(n_vocab)
m = model.to(device)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print(f"Step {iter} / {max_iters}", end='\r')


    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['dev']:.4f}, val accuracy = {math.exp(-(losses['dev']))*100:4f}%")


    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))