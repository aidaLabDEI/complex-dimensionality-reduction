import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import torch

randd = np.random
randd.seed(42)



def exec_jl(i, k):
    randd.seed(i)
    h = np.zeros(k)
    for j in range(k):
        A = (1 - 2*randd.randint(0, 2, (1,d), bool))

        gx = A @ (v * v)
        gw = A @ (w * w)

        h[j] = gx * gw
        
    return h


def exec_block(i, kL, L):
    randd.seed(i)
    h = np.zeros( (L, kL//L) )
    for ell in range(L):
        for j in range(kL//L):
            A = ((1+1j) + (-1+1j)*(1 - 2*randd.randint(0, 2, (1,d//L), bool)))*(1 - 2*randd.randint(0, 2, (1,d//L), bool))/2 #complex JL
            A2 = A * A

            gx = A @ v[(d//L)*ell:(d//L)*(ell+1)] 
            gw = A2 @ (w[(d//L)*ell:(d//L)*(ell+1)] * w[(d//L)*ell:(d//L)*(ell+1)])

            h[ell][j] = np.real(gx * gx * gw)
    return h



experiment = int(sys.argv[1])

d = 20000
N = 50

w = np.ones(d) + 0.2*( np.random.rand(d) ) 
v = 1-2*np.random.rand(d)
v = 1*(v/np.linalg.norm(v)) # norm is 1
print(w[0:10])
print(v[0:10])




if experiment == 0:
    siz = [100, 1000, 10000]
    k = 10000
    print(np.linalg.norm(v*w))
    h2 = [ exec_jl(i, k) for i in range(N) ]
    out = [np.linalg.norm(v*w)]
    for s in siz:
        norm2 = [0 for i in range(N)]
        for i in range(N):
            norm2[i] = np.sum(h2[i][0:s])/s
        print(norm2)
        out.append(norm2)
    print()
    torch.save(out, f"jl_norm={np.linalg.norm(v*w)}_N={N}.pt")

if experiment == 1:
    siz = [100, 1000, 10000]
    k = 10000
    L = 1
    print(np.linalg.norm(v*w))
    h2 = [ exec_block(i, kL=k, L=L) for i in range(N) ]
    out = [np.linalg.norm(v*w)]
    for s in siz:
        norm2 = [0 for i in range(N)]
        for i in range(N):
            for ell in range(L):           
                norm2[i] += np.sum(h2[i][ell][0 :(s//L)])/ (s//L)
        print(norm2)
        out.append(norm2)
    torch.save(out, f"block={L}_norm={np.linalg.norm(v*w)}_N={N}.pt")


if experiment == 2:
    siz = [100, 1000, 10000]
    k = 10000
    L = 100
    print(np.linalg.norm(v*w))
    h2 = [ exec_block(i, kL=k, L=L) for i in range(N) ]
    out = [np.linalg.norm(v*w)]
    for s in siz:
        norm2 = [0 for i in range(N)]
        for i in range(N):
            for ell in range(L):           
                norm2[i] += np.sum(h2[i][ell][0 :(s//L)])/ (s//L)
        print(norm2)
        out.append(norm2)
    torch.save(out, f"block={L}_norm={np.linalg.norm(v*w)}_N={N}.pt")

if experiment == 3:
    siz = [100, 1000, 10000]
    k = 10000
    h2 = [ [ exec_block(i, kL=Li, L=Li) for Li in siz] for i in range(N) ]
    out = [np.linalg.norm(v*w)]
    for si, s in enumerate(siz):
        norm2 = [0 for i in range(N)]
        for i in range(N):
            for ell in range(s):           
                norm2[i] += np.sum(h2[i][si][ell][0 :1])
        print(norm2)
        out.append(norm2)
    torch.save(out, f"block=max_norm={np.linalg.norm(v*w)}_N={N}.pt")
    
# sparse
if experiment == 4:
    m = 200 # sparse entries
    sparse = np.zeros(d, dtype=int)
    sparse[np.random.choice(d, m, replace=False)] = 1
    w = w * sparse 
    v = v * sparse
    v = 1*(v/np.linalg.norm(v)) # norm is 1

    siz = [100, 1000, 10000]
    k = 10000
    L = 1
    print(np.linalg.norm(v*w))
    h2 = [ exec_block(i, kL=k, L=L) for i in range(N) ]
    out = [np.linalg.norm(v*w)]
    for s in siz:
        norm2 = [0 for i in range(N)]
        for i in range(N):
            for ell in range(L):           
                norm2[i] += np.sum(h2[i][ell][0 :(s//L)])/ (s//L)
        print(norm2)
        out.append(norm2)
    torch.save(out, f"sparse_block={L}_norm={np.linalg.norm(v*w)}_N={N}.pt")