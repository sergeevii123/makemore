import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


words = open("data/names.txt").read().split()
chars  = sorted(list(set("".join(words))))
stoi = {char:i+1 for i,char in enumerate(chars)}
stoi["."] = 0
itos = {i:char for char,i in stoi.items()}


def build_dataset(words):
    block_size = 3 
    X,Y = [],[]
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] #crop and append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X,Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xva, Yva = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,20), generator=g)
W1 = torch.randn((20*3,200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200,27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True


for i in range(200000):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (64,))
    #forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1,20*3) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    
    #backward pass
    for p in parameters: 
        p.grad = None
    loss.backward()
    # lr = lrs[i]
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters: 
        p.data += -lr * p.grad


emb = C[Xtr]
h = torch.tanh(emb.view(-1,20*3) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print(f"Train loss: {loss.item():.4f}")

emb = C[Xva]
h = torch.tanh(emb.view(-1,20*3) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yva)
print(f"Validation loss: {loss.item():.4f}")

emb = C[Xte]
h = torch.tanh(emb.view(-1,20*3) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yte)
print(f"Test loss: {loss.item():.4f}")

print("Sampling:")
# sample from the model
g = torch.Generator().manual_seed(2147483647+10)

for _ in range(20):
    out = []
    context = [0] * 3
    while True:
        emb = C[torch.tensor(context)]
        h = torch.tanh(emb.view(1,-1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:]  + [ix]
        out.append(ix)
        if ix == 0: break

    print(''.join(itos[i] for i in out))

