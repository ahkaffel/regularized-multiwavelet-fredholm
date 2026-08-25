#!/usr/bin/env python3
"""Independent verification of the six integration benchmarks in the manuscript."""
import csv, math
from pathlib import Path
import numpy as np
import mpmath as mp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'integration_benchmarks_verified.csv'

HYB={3:([3,2,3],8),4:([13,11,11,13],48),5:([275,100,402,100,275],1152),6:([247,139,254,254,139,247],1280)}

def midpoint_1d(f,a,b,N):
    x=a+(np.arange(N)+0.5)*(b-a)/N
    return (b-a)*np.mean(f(x))

def midpoint_2d(f,a,b,c,d,N):
    x=a+(np.arange(N)+0.5)*(b-a)/N
    y=c+(np.arange(N)+0.5)*(d-c)/N
    return (b-a)*(d-c)*np.mean(f(x[:,None],y[None,:]))

def midpoint_sum3(g,a,b,N):
    h=(b-a)/N
    counts=np.convolve(np.convolve(np.ones(N),np.ones(N)),np.ones(N))
    sums=3*a+(np.arange(counts.size)+1.5)*h
    return h**3*np.sum(counts*g(sums))

def hybrid_nodes_weights(m,n,a,b):
    ww,den=HYB[m]; xs=[]; ws=[]
    for i in range(1,n+1):
        for r,w in enumerate(ww):
            num=2*m*i-(2*m-(2*r+1)); t=num/(2*m*n)
            xs.append(a+(b-a)*t); ws.append((b-a)*w/(den*n))
    return np.asarray(xs,float),np.asarray(ws,float)

def hybrid_1d(f,a,b,m,n):
    x,w=hybrid_nodes_weights(m,n,a,b); return np.sum(w*f(x))

def hybrid_2d(f,a,b,c,d,m,n):
    x,wx=hybrid_nodes_weights(m,n,a,b); y,wy=hybrid_nodes_weights(m,n,c,d)
    return np.sum(wx[:,None]*wy[None,:]*f(x[:,None],y[None,:]))

def hybrid_sum3(g,a,b,m,n):
    x,w=hybrid_nodes_weights(m,n,a,b)
    # The nodes are equally spaced; convolving weights avoids forming a 3D array.
    wc=np.convolve(np.convolve(w,w),w)
    step=x[1]-x[0]; sums=3*x[0]+np.arange(wc.size)*step
    return np.sum(wc*g(sums))

mp.mp.dps=50
I1=float(mp.quad(lambda t: mp.sin(t*t),[0,1]))
I2=float(mp.quad(lambda t: mp.sqrt(t*t-5*t+31),[0,5]))
I3=float(mp.e**(-1))
I4=2.0 # integral of sin(x+y) over [0,pi/2]^2
I5=2*math.log(1+math.sqrt(2))
F=lambda s:(8/15)*s**2.5
I6=F(6)-3*F(5)+3*F(4)-F(3)

examples={
1:(lambda x:np.sin(x*x),(0,1),I1),
2:(lambda x:np.sqrt(x*x-5*x+31),(0,5),I2),
3:(lambda x:np.where(x==0,0.0,np.exp(-1/x)/(x*x)),(0,1),I3),
}
rows=[]
for ex,(f,(a,b),I) in examples.items():
    for lev in range(4,8):
        q=midpoint_1d(f,a,b,2**(lev+1)); rows.append((ex,'Haar',f'j={lev}',abs(q-I)/abs(I)))
        q=midpoint_1d(f,a,b,2**(lev+2)); rows.append((ex,'LLMW',f'k={lev}',abs(q-I)/abs(I)))
    for m,n in [(3,5),(4,8),(5,12),(6,20)]:
        q=hybrid_1d(f,a,b,m,n); rows.append((ex,'Hybrid',f'm={m}, n={n}',abs(q-I)/abs(I)))

# Example 4: [0,pi/2]^2, exact 2.
f4=lambda x,y:np.sin(x+y)
# Example 5: integrable point singularity; midpoint/hybrid nodes avoid the origin.
f5=lambda x,y:1/np.sqrt(x*x+y*y)
for ex,f,bounds,I in [(4,f4,(0,math.pi/2,0,math.pi/2),I4),(5,f5,(0,1,0,1),I5)]:
    a,b,c,d=bounds
    for lev in range(4,8):
        q=midpoint_2d(f,a,b,c,d,2**(lev+1)); rows.append((ex,'Haar',f'j={lev}',abs(q-I)/abs(I)))
        q=midpoint_2d(f,a,b,c,d,2**(lev+2)); rows.append((ex,'LLMW',f'k={lev}',abs(q-I)/abs(I)))
    for m,n in [(3,5),(4,8),(5,12),(6,20)]:
        q=hybrid_2d(f,a,b,c,d,m,n); rows.append((ex,'Hybrid',f'm={m}, n={n}',abs(q-I)/abs(I)))

# Example 6: repeated integration gives (8/15)[6^(5/2)-3*5^(5/2)+3*4^(5/2)-3^(5/2)].
g6=lambda s:1/np.sqrt(s)
for lev in range(4,8):
    q=midpoint_sum3(g6,1,2,2**(lev+1)); rows.append((6,'Haar',f'j={lev}',abs(q-I6)/abs(I6)))
    q=midpoint_sum3(g6,1,2,2**(lev+2)); rows.append((6,'LLMW',f'k={lev}',abs(q-I6)/abs(I6)))
for m,n in [(3,5),(4,8),(5,12),(6,20)]:
    q=hybrid_sum3(g6,1,2,m,n); rows.append((6,'Hybrid',f'm={m}, n={n}',abs(q-I6)/abs(I6)))

with OUT.open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['example','method','parameter','relative_error']); w.writerows(rows)
print(OUT)
