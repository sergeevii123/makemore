import torch

words = open("data/names.txt").read().split()

N = torch.zeros((27,27), dtype = torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1

P = (N+1).float()
# bigram probabilities
P /= P.sum(dim=1, keepdim = True)


log_likelihood = 0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
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
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

