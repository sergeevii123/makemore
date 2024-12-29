import torch
import torch.nn.functional as F

words = open("data/names.txt").read().split()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# create dataset
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# init the network
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True) 

for k in range(200):
    # forward pass
    xenc = F.one_hot(xs, num_classes = 27).float()
    logits = (xenc @ W)
    counts = logits.exp() # ~N
    probs = counts / counts.sum(dim=1, keepdim = True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    # backward pass
    W.grad = None # set grad to zero
    loss.backward()
    # update
    W.data += -50 * W.grad


log_likelihood = 0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xenc = F.one_hot(torch.tensor([ix1]), num_classes = 27).float()
        logits = (xenc @ W) # predict log counts
        counts = logits.exp() # counts, ~N
        p = counts / counts.sum(dim=1, keepdim = True) # probs
        logprob = torch.log(p[0,ix2])
        log_likelihood += logprob
        n+=1

print(f"Log likelihood: {log_likelihood:.4f}")
nll = -log_likelihood
print(f"Negative log likelihood: {nll:.4f}")
print(f"Average negative log likelihood: {nll/n:.4f}")
print()
print("Sampling:")
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    
    out = []
    ix = 0
    while True:
        # bigram
        # p = P[ix]
        
        # neural net
        xenc = F.one_hot(torch.tensor([ix]), num_classes = 27).float()
        logits = (xenc @ W) # predict log counts
        counts = logits.exp() # counts, ~N
        p = counts / counts.sum(dim=1, keepdim = True) # probs

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))