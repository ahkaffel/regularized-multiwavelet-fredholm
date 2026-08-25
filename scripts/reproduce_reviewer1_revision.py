#!/usr/bin/env python3
"""Reproduce the numerical diagnostics added for Reviewer 1.

The script implements a common, basis-normalized collocation/Tikhonov protocol
for five approximation spaces and a classical Nyström baseline.  It generates
CSV tables and publication figures used in the revised manuscript.
"""
from __future__ import annotations
import csv, json, math, platform, time
from pathlib import Path
import numpy as np
import scipy
from scipy.special import eval_legendre
from scipy.interpolate import BSpline
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer1"
FIG = ROOT / "figures" / "reviewer1"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
N = 32
TAU = 1.01
SEEDS = list(range(1000, 1030))
NOISE_LEVELS = [1e-4, 1e-3, 1e-2, 5e-2]

# ---------- quadrature ----------
def composite_gauss(npanels=256, order=8):
    q, w = leggauss(order)
    xs, ws = [], []
    for p in range(npanels):
        a, b = p / npanels, (p + 1) / npanels
        xs.append((a + b) / 2 + (b - a) / 2 * q)
        ws.append((b - a) / 2 * w)
    return np.concatenate(xs), np.concatenate(ws)

QX, QW = composite_gauss()

# ---------- basis definitions ----------
def haar_basis(x, N=N):
    x = np.asarray(x)
    B = np.zeros((x.size, N)); B[:, 0] = 1.0
    idx = 1; j = 0
    while idx < N:
        for k in range(2**j):
            if idx >= N: break
            a, mid, b = k/2**j, (k+0.5)/2**j, (k+1)/2**j
            m1 = (x >= a) & (x < mid)
            m2 = (x >= mid) & ((x < b) | ((k == 2**j-1) & (x <= b)))
            B[m1, idx] = 2**(j/2); B[m2, idx] = -2**(j/2)
            idx += 1
        j += 1
    return B

def legendre_wavelet_basis(x, N=N, klevel=3):
    x = np.asarray(x); cells = 2**(klevel-1); M = N // cells
    B = np.zeros((x.size, N)); col = 0
    for n in range(1, cells+1):
        a, b = (n-1)/cells, n/cells
        mask = (x >= a) & ((x < b) | ((n == cells) & (x <= b)))
        xi = 2*cells*x[mask] - 2*n + 1
        for m in range(M):
            B[mask, col] = np.sqrt(m + 0.5) * (2**(klevel/2)) * eval_legendre(m, xi)
            col += 1
    return B

def hybrid_basis(x, N=N, ncells=8):
    x = np.asarray(x); m = N // ncells
    B = np.zeros((x.size, N)); col = 0
    for i in range(1, ncells+1):
        a, b = (i-1)/ncells, i/ncells
        mask = (x >= a) & ((x < b) | ((i == ncells) & (x <= b)))
        xi = 2*ncells*x[mask] - 2*i + 1
        for j in range(m):
            B[mask, col] = np.sqrt(ncells*(2*j+1)) * eval_legendre(j, xi)
            col += 1
    return B

def bspline_basis(x, N=N, degree=3):
    x = np.asarray(x)
    interior_count = N - degree - 1
    interior = np.linspace(0, 1, interior_count+2)[1:-1] if interior_count > 0 else np.array([])
    knots = np.r_[np.zeros(degree+1), interior, np.ones(degree+1)]
    return BSpline.design_matrix(x, knots, degree, extrapolate=False).toarray()

def llmw_basis(x, N=N):
    x = np.asarray(x)
    B = [np.ones_like(x), np.sqrt(3)*(2*x-1)]
    def psi0(u):
        out = np.zeros_like(u)
        m1=(u>=0)&(u<0.5); m2=(u>=0.5)&(u<=1)
        out[m1] = -np.sqrt(3)*(4*u[m1]-1)
        out[m2] =  np.sqrt(3)*(4*u[m2]-3)
        return out
    def psi1(u):
        out = np.zeros_like(u)
        m1=(u>=0)&(u<0.5); m2=(u>=0.5)&(u<=1)
        out[m1] = 6*u[m1]-1; out[m2] = 6*u[m2]-5
        return out
    level = 0
    while len(B) < N:
        for n in range(2**level):
            u = 2**level*x - n
            if n < 2**level-1:
                u = np.where(np.isclose(u,1.0,atol=1e-15),2.0,u)
            fac = 2**(level/2)
            B.append(fac*psi0(u))
            if len(B) >= N: break
            B.append(fac*psi1(u))
            if len(B) >= N: break
        level += 1
    return np.column_stack(B[:N])

BASIS = {
    "Haar": haar_basis,
    "Hybrid": hybrid_basis,
    "Legendre": legendre_wavelet_basis,
    "B-spline": bspline_basis,
    "LLMW": llmw_basis,
}

# ---------- common collocation / normalization ----------
def mass_whitener(Bq):
    M = Bq.T @ (QW[:,None] * Bq)
    lam, Q = np.linalg.eigh(M)
    if lam.min() <= 0: raise RuntimeError("non-positive Gram matrix")
    Minvhalf = Q @ np.diag(1/np.sqrt(lam)) @ Q.T
    return M, Minvhalf

def assemble(basis_fn, kernel, N=N):
    x = (np.arange(N)+0.5)/N
    Bq = basis_fn(QX)
    M, Minvhalf = mass_whitener(Bq)
    K = kernel(x[:,None], QX[None,:])
    A = K @ (QW[:,None]*Bq)
    return x, Bq, M, Minvhalf, A, A @ Minvhalf

def exact_data(kernel, f, x):
    K = kernel(x[:,None], QX[None,:])
    return K @ (QW * f(QX))

def solve_morozov(U, s, Vt, gd, eps, tau=TAU):
    target = tau*eps
    ug = U.T @ gd
    def residual(alpha):
        z = Vt.T @ ((s/(s*s+alpha))*ug)
        return np.linalg.norm(U @ (s*(Vt@z)) - gd)
    lo, hi = 1e-16, 1e4
    rlo, rhi = residual(lo), residual(hi)
    if target <= rlo: alpha = lo
    elif target >= rhi: alpha = hi
    else:
        alpha = 10**brentq(lambda q: residual(10**q)-target, -16, 4, xtol=1e-12, rtol=1e-12)
    z = Vt.T @ ((s/(s*s+alpha))*ug)
    return alpha, z, residual(alpha)

def reconstruct(basis_fn, Minvhalf, z, x):
    return basis_fn(x) @ (Minvhalf @ z)

GAUSSIAN = lambda sigma: (lambda x,y: np.exp(-((x-y)**2)/(sigma**2)))
F_SMOOTH = lambda x: np.sin(np.pi*x) + x*(1-x)
F_NONSMOOTH = lambda x: np.abs(x-0.5)

# ---------- statistical diagnostics ----------
def statistical_rows(sigma, f, noise_levels, seeds=SEEDS):
    rows=[]; xeval=np.linspace(0,1,2001); fex=f(xeval); fden=np.linalg.norm(fex)
    ker=GAUSSIAN(sigma)
    for name, fn in BASIS.items():
        x, Bq, M, Minvhalf, A, Aw = assemble(fn, ker)
        U,s,Vt=np.linalg.svd(Aw,full_matrices=False)
        g=exact_data(ker,f,x)
        for delta in noise_levels:
            vals=[]
            for seed in seeds:
                rng=np.random.default_rng(seed)
                e=rng.standard_normal(g.size)
                e=e/np.linalg.norm(e)*(delta*np.linalg.norm(g))
                gd=g+e
                alpha,z,res=solve_morozov(U,s,Vt,gd,np.linalg.norm(e))
                frec=reconstruct(fn,Minvhalf,z,xeval)
                es=np.linalg.norm(frec-fex)/fden
                er=np.linalg.norm(Aw@z-gd)/np.linalg.norm(gd)
                kreg=(s[0]**2+alpha)/(s[-1]**2+alpha)
                vals.append((alpha,es,er,kreg))
            arr=np.asarray(vals)
            mean=arr.mean(0); sd=arr.std(0,ddof=1); ci=1.96*sd/np.sqrt(len(seeds))
            rows.append({
                "basis":name,"sigma":sigma,"delta":delta,"runs":len(seeds),
                "alpha_mean":mean[0],"alpha_sd":sd[0],"alpha_ci95":ci[0],
                "Esol_mean":mean[1],"Esol_sd":sd[1],"Esol_ci95":ci[1],
                "Eres_mean":mean[2],"Eres_sd":sd[2],"Eres_ci95":ci[2],
                "kreg_mean":mean[3],"kreg_sd":sd[3],"kreg_ci95":ci[3],
                "sigma_max":s[0],"sigma_min":s[-1],"kappa_A":s[0]/s[-1],
            })
    return rows

def write_csv(path, rows):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("Running 30-realization noise study...")
SMOOTH = statistical_rows(0.1,F_SMOOTH,NOISE_LEVELS)
NONS = statistical_rows(0.1,F_NONSMOOTH,[1e-3])
STRONG = statistical_rows(0.2,F_SMOOTH,[1e-3])
write_csv(OUT/"gaussian_smooth_statistics.csv",SMOOTH)
write_csv(OUT/"gaussian_nonsmooth_statistics.csv",NONS)
write_csv(OUT/"gaussian_sigma02_statistics.csv",STRONG)

# ---------- classical Nyström baseline ----------
def nystrom_stats(sigma=0.1,delta=1e-3,seeds=SEEDS):
    n=N; y=(np.arange(n)+0.5)/n; x=y.copy(); w=1/n
    A=GAUSSIAN(sigma)(x[:,None],y[None,:])*w
    Minvhalf=np.eye(n)*np.sqrt(n); Aw=A@Minvhalf
    U,s,Vt=np.linalg.svd(Aw,full_matrices=False)
    g=exact_data(GAUSSIAN(sigma),F_SMOOTH,x)
    xeval=np.linspace(0,1,2001); fex=F_SMOOTH(xeval)
    vals=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); e=rng.standard_normal(n)
        e=e/np.linalg.norm(e)*(delta*np.linalg.norm(g)); gd=g+e
        a,z,r=solve_morozov(U,s,Vt,gd,np.linalg.norm(e))
        c=Minvhalf@z; ind=np.minimum((xeval*n).astype(int),n-1); frec=c[ind]
        vals.append((a,np.linalg.norm(frec-fex)/np.linalg.norm(fex),np.linalg.norm(Aw@z-gd)/np.linalg.norm(gd)))
    arr=np.asarray(vals); mean=arr.mean(0); sd=arr.std(0,ddof=1); ci=1.96*sd/np.sqrt(len(seeds))
    return {"method":"Nyström-midpoint","alpha_mean":mean[0],"Esol_mean":mean[1],"Esol_ci95":ci[1],"Eres_mean":mean[2],"kappa_A":s[0]/s[-1]}
NY=nystrom_stats()
with open(OUT/"nystrom_baseline.json","w") as f: json.dump(NY,f,indent=2)

# ---------- corrected quadrature verification ----------
import mpmath as mp
mp.mp.dps=50
def hybrid_rule(f,m,n,a=0,b=1):
    weights={3:([3,2,3],8),4:([13,11,11,13],48),5:([275,100,402,100,275],1152),6:([247,139,254,254,139,247],1280)}
    ww,den=weights[m]; s=mp.mpf('0')
    for i in range(1,n+1):
        for r,wgt in enumerate(ww):
            num=2*m*i-(2*m-(2*r+1)); t=mp.mpf(num)/(2*m*n); x=a+(b-a)*t
            s += wgt*f(x)
    return (b-a)*s/(den*n)
def midpoint_rule(f,a,b,n):
    return (b-a)/n*sum(f(a+(b-a)*(k+mp.mpf('0.5'))/n) for k in range(n))
qt=[]
examples=[
    ("Example 1",lambda x:mp.sin(x*x),mp.quad(lambda x:mp.sin(x*x),[0,1]),0,1),
    ("Example 2",lambda x:mp.sqrt(x*x-5*x+31),mp.quad(lambda x:mp.sqrt(x*x-5*x+31),[0,5]),0,5),
    ("Example 3",lambda x: mp.e**(-1/x)/(x*x) if x != 0 else mp.mpf('0'),mp.e**(-1),0,1),
]
for ename,f,I,a,b in examples:
    for level in range(4,8):
        q=midpoint_rule(f,a,b,2**(level+1)); qt.append({"example":ename,"method":"Haar","param":f"j={level}","rel_error":float(abs(q-I)/abs(I))})
        q=midpoint_rule(f,a,b,2**(level+2)); qt.append({"example":ename,"method":"LLMW","param":f"k={level}","rel_error":float(abs(q-I)/abs(I))})
    for m,n in [(3,5),(4,8),(5,12),(6,20)]:
        q=hybrid_rule(f,m,n,a,b); qt.append({"example":ename,"method":"Hybrid","param":f"m={m}, n={n}","rel_error":float(abs(q-I)/abs(I))})
write_csv(OUT/"corrected_quadrature_errors.csv",qt)

# ---------- benchmark whole-domain verification (replaces pointwise superiority claims) ----------
def benchmark_stats(kernel,f,gfun,delta=1e-6,seeds=range(1000,1020)):
    rows=[]; xeval=np.linspace(0,1,2001); fex=f(xeval); den=np.linalg.norm(fex)
    for name,fn in BASIS.items():
        x,Bq,M,Minvhalf,A,Aw=assemble(fn,kernel); U,s,Vt=np.linalg.svd(Aw,full_matrices=False); g=gfun(x)
        vals=[]
        for seed in seeds:
            rng=np.random.default_rng(seed); e=rng.standard_normal(g.size); e=e/np.linalg.norm(e)*delta*np.linalg.norm(g); gd=g+e
            a,z,r=solve_morozov(U,s,Vt,gd,np.linalg.norm(e)); frec=reconstruct(fn,Minvhalf,z,xeval)
            vals.append((np.linalg.norm(frec-fex)/den,np.max(np.abs(frec-fex)),a))
        arr=np.asarray(vals); mean=arr.mean(0); sd=arr.std(0,ddof=1); ci=1.96*sd/np.sqrt(len(seeds))
        rows.append({"basis":name,"delta":delta,"runs":len(seeds),"L2_mean":mean[0],"L2_ci95":ci[0],"Linf_mean":mean[1],"Linf_ci95":ci[1],"alpha_mean":mean[2],"kappa_A":s[0]/s[-1]})
    return rows
K1=lambda x,y:np.sin(x*y); F1=lambda y:y
G1=lambda x:np.where(np.abs(x)>1e-14,(np.sin(x)-x*np.cos(x))/(x*x),0.5)
K2=lambda x,y:np.exp((x**2)*y); F2=lambda y:np.exp(y); G2=lambda x:(np.exp(x*x+1)-1)/(x*x+1)
B1=benchmark_stats(K1,F1,G1); B2=benchmark_stats(K2,F2,G2)
write_csv(OUT/"fredholm_problem1_norm_errors.csv",B1); write_csv(OUT/"fredholm_problem2_norm_errors.csv",B2)

# ---------- GCV cross-check ----------
def gcv_crosscheck(name="LLMW",sigma=0.1,delta=1e-3,seed=1000):
    fn=BASIS[name]; ker=GAUSSIAN(sigma); x,Bq,M,Minvhalf,A,Aw=assemble(fn,ker); U,s,Vt=np.linalg.svd(Aw,full_matrices=False)
    g=exact_data(ker,F_SMOOTH,x); rng=np.random.default_rng(seed); e=rng.standard_normal(N); e=e/np.linalg.norm(e)*delta*np.linalg.norm(g); gd=g+e
    am,zm,_=solve_morozov(U,s,Vt,gd,np.linalg.norm(e)); ug=U.T@gd
    grid=np.logspace(-12,1,800); gcv=[]
    for a in grid:
        z=Vt.T@((s/(s*s+a))*ug); r=np.linalg.norm(Aw@z-gd); trH=np.sum(s*s/(s*s+a)); gcv.append(r*r/(N-trH)**2)
    ag=grid[int(np.argmin(gcv))]
    xev=np.linspace(0,1,2001); fex=F_SMOOTH(xev)
    def err(a):
        z=Vt.T@((s/(s*s+a))*ug); fr=reconstruct(fn,Minvhalf,z,xev); return np.linalg.norm(fr-fex)/np.linalg.norm(fex)
    return {"basis":name,"delta":delta,"seed":seed,"alpha_morozov":am,"Esol_morozov":err(am),"alpha_GCV":ag,"Esol_GCV":err(ag)}
GC=gcv_crosscheck()
with open(OUT/"parameter_choice_crosscheck.json","w") as f: json.dump(GC,f,indent=2)

# ---------- figures ----------
def rows_for(rows,basis,delta=None):
    rr=[r for r in rows if r['basis']==basis and (delta is None or r['delta']==delta)]
    return sorted(rr,key=lambda z:z['delta'])

# error vs noise with 95% CI
plt.figure(figsize=(7.2,4.8))
for name in BASIS:
    rr=rows_for(SMOOTH,name); xs=np.array([r['delta'] for r in rr]); ys=np.array([r['Esol_mean'] for r in rr]); ci=np.array([r['Esol_ci95'] for r in rr])
    plt.errorbar(xs,ys,yerr=ci,marker='o',capsize=3,label=name)
plt.xscale('log'); plt.yscale('log'); plt.xlabel('Relative noise level $\\delta$'); plt.ylabel('Relative solution error'); plt.grid(True,which='both',alpha=.25); plt.legend(ncol=2); plt.tight_layout(); plt.savefig(FIG/'fig_error_vs_noise_ci.png',dpi=220); plt.close()

# alpha vs noise
plt.figure(figsize=(7.2,4.8))
for name in BASIS:
    rr=rows_for(SMOOTH,name); xs=np.array([r['delta'] for r in rr]); ys=np.array([r['alpha_mean'] for r in rr]); ci=np.array([r['alpha_ci95'] for r in rr])
    plt.errorbar(xs,ys,yerr=ci,marker='o',capsize=3,label=name)
plt.xscale('log'); plt.yscale('log'); plt.xlabel('Relative noise level $\\delta$'); plt.ylabel('Morozov-selected $\\alpha$'); plt.grid(True,which='both',alpha=.25); plt.legend(ncol=2); plt.tight_layout(); plt.savefig(FIG/'fig_alpha_vs_noise_ci.png',dpi=220); plt.close()

# condition numbers at delta=1e-3
plt.figure(figsize=(7.2,4.8)); names=list(BASIS.keys())
kA=[]; kR=[]
for name in names:
    r=rows_for(SMOOTH,name,1e-3)[0]; kA.append(r['kappa_A']); kR.append(r['kreg_mean'])
xp=np.arange(len(names)); width=.36
plt.bar(xp-width/2,kA,width,label='$\\kappa_2(A_N)$'); plt.bar(xp+width/2,kR,width,label='$\\kappa_2(A_N^TA_N+\\alpha I)$'); plt.yscale('log'); plt.xticks(xp,names,rotation=20); plt.ylabel('Condition number'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'fig_conditioning_recomputed.png',dpi=220); plt.close()

# singular spectra for sigma .1 and .2 using fine Nystrom matrix
plt.figure(figsize=(7.2,4.8)); nsv=160; y=(np.arange(nsv)+.5)/nsv
for sig in [0.1,0.2]:
    A=GAUSSIAN(sig)(y[:,None],y[None,:])/nsv; ss=np.linalg.svd(A,compute_uv=False); plt.semilogy(np.arange(1,len(ss)+1),ss/ss[0],label=f'$\\sigma={sig}$')
plt.xlabel('Singular-value index'); plt.ylabel('Normalized singular value'); plt.grid(True,which='both',alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'fig_gaussian_singular_decay.png',dpi=220); plt.close()

# representative reconstructions (seed 1000, delta1e-3)
def recon_plot(f,filename):
    ker=GAUSSIAN(.1); xev=np.linspace(0,1,1200); plt.figure(figsize=(7.2,4.8)); plt.plot(xev,f(xev),linewidth=2.4,label='Exact')
    for name,fn in BASIS.items():
        x,Bq,M,Minvhalf,A,Aw=assemble(fn,ker); U,s,Vt=np.linalg.svd(Aw,full_matrices=False); g=exact_data(ker,f,x); rng=np.random.default_rng(1000); e=rng.standard_normal(N); e=e/np.linalg.norm(e)*(1e-3*np.linalg.norm(g)); a,z,_=solve_morozov(U,s,Vt,g+e,np.linalg.norm(e)); plt.plot(xev,reconstruct(fn,Minvhalf,z,xev),linewidth=1.1,label=name)
    plt.xlabel('$y$'); plt.ylabel('$f(y)$'); plt.legend(ncol=2); plt.tight_layout(); plt.savefig(FIG/filename,dpi=220); plt.close()
recon_plot(F_SMOOTH,'fig_reconstruction_smooth_recomputed.png'); recon_plot(F_NONSMOOTH,'fig_reconstruction_nonsmooth_recomputed.png')

# L-curve with Morozov marker for LLMW
fn=LLMW=llmw_basis; ker=GAUSSIAN(.1); x,Bq,M,Minvhalf,A,Aw=assemble(fn,ker); U,s,Vt=np.linalg.svd(Aw,full_matrices=False); g=exact_data(ker,F_SMOOTH,x); rng=np.random.default_rng(1000); e=rng.standard_normal(N); e=e/np.linalg.norm(e)*(1e-3*np.linalg.norm(g)); gd=g+e; am,zm,_=solve_morozov(U,s,Vt,gd,np.linalg.norm(e)); ug=U.T@gd
grid=np.logspace(-10,0,240); rr=[]; zz=[]
for a in grid:
    z=Vt.T@((s/(s*s+a))*ug); rr.append(np.linalg.norm(Aw@z-gd)); zz.append(np.linalg.norm(z))
rm=np.linalg.norm(Aw@zm-gd); nm=np.linalg.norm(zm)
plt.figure(figsize=(6.2,5)); plt.loglog(rr,zz,'-'); plt.scatter([rm],[nm],s=55,label='Morozov $\\alpha$'); plt.xlabel('Residual norm'); plt.ylabel('Coefficient norm'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'fig_lcurve_morozov_marked.png',dpi=220); plt.close()

# ---------- timing benchmark ----------
def benchmark_time(fn,repeats_assembly=25,repeats_solve=250):
    ker=GAUSSIAN(.1); ta=[]
    for _ in range(repeats_assembly):
        t=time.perf_counter(); out=assemble(fn,ker); ta.append(time.perf_counter()-t)
    x,Bq,M,Minvhalf,A,Aw=out; g=exact_data(ker,F_SMOOTH,x); ts=[]
    for _ in range(repeats_solve):
        t=time.perf_counter(); U,s,Vt=np.linalg.svd(Aw,full_matrices=False); a=6e-4; _=Vt.T@((s/(s*s+a))*(U.T@g)); ts.append(time.perf_counter()-t)
    return float(np.median(ta)),float(np.median(ts))
timing=[]
for name,fn in BASIS.items():
    a,s=benchmark_time(fn); timing.append({"basis":name,"assembly_median_s":a,"solve_median_s":s,"total_median_s":a+s})
write_csv(OUT/"timing_repeated_medians.csv",timing)
plt.figure(figsize=(7.2,4.8)); names=[r['basis'] for r in timing]; vals=[1000*r['total_median_s'] for r in timing]; plt.bar(np.arange(len(names)),vals); plt.xticks(np.arange(len(names)),names,rotation=20); plt.ylabel('Median assembly + solve time (ms)'); plt.tight_layout(); plt.savefig(FIG/'fig_cpu_cost_recomputed.png',dpi=220); plt.close()

# Environment metadata
meta={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"platform":platform.platform(),"N":N,"tau":TAU,"seeds":SEEDS,"noise_levels":NOISE_LEVELS,"quadrature":"256 panels x 8-point Gauss-Legendre"}
with open(OUT/"environment.json","w") as f: json.dump(meta,f,indent=2)

print("Done. Outputs written to",OUT)
