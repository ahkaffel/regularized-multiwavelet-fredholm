#!/usr/bin/env python3
"""Additional diagnostics requested by Reviewer 2.

Run after (or independently of) reproduce_reviewer1_revision.py.  The script
reuses the common basis-normalized Tikhonov--Morozov implementation and adds:
(i) an oscillatory smooth-kernel problem; (ii) a Hilbert--Schmidt weakly
singular kernel; (iii) observed convergence-rate tables; (iv) alpha-sensitivity
curves; and (v) repeated CPU timing for the added inverse tests.
"""
from __future__ import annotations
import csv, math, runpy, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_jacobi

ROOT = Path(__file__).resolve().parents[1]
R1 = runpy.run_path(str(ROOT / 'scripts' / 'reproduce_reviewer1_revision.py'))
OUT = ROOT / 'results' / 'reviewer2'; FIG = ROOT / 'figures' / 'reviewer2'
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

QX,QW = R1['QX'],R1['QW']; mass_whitener=R1['mass_whitener']; solve_morozov=R1['solve_morozov']
haar_basis=R1['haar_basis']; llmw_basis=R1['llmw_basis']; hybrid_basis=R1['hybrid_basis']; legendre_wavelet_basis=R1['legendre_wavelet_basis']; bspline_basis=R1['bspline_basis']
F = R1['F_SMOOTH']; TAU=R1['TAU']
N0=32; SEEDS=list(range(2000,2020)); DELTA=1e-3

def bmat(name,x,N):
    if name=='Haar': return haar_basis(x,N=N)
    if name=='LLMW': return llmw_basis(x,N=N)
    if name=='B-spline': return bspline_basis(x,N=N,degree=3)
    if name=='Hybrid':
        nc=max(1,N//4)
        return hybrid_basis(x,N=N,ncells=nc)
    if name=='Legendre':
        cells=max(1,N//4); klevel=int(round(math.log2(cells)))+1
        return legendre_wavelet_basis(x,N=N,klevel=klevel)
    raise KeyError(name)
NAMES=['Haar','Hybrid','Legendre','B-spline','LLMW']

def assemble_generic(name,kernel,N=N0):
    x=(np.arange(N)+0.5)/N; Bq=bmat(name,QX,N); M,Minvhalf=mass_whitener(Bq)
    K=kernel(x[:,None],QX[None,:]); Ac=K @ (QW[:,None]*Bq); return x,Minvhalf,Ac@Minvhalf

def data_generic(kernel,f,x): return kernel(x[:,None],QX[None,:]) @ (QW*f(QX))

def reconstruct(name,N,Minvhalf,z,x): return bmat(name,x,N) @ (Minvhalf@z)

def stats_generic(kernel,f,label,N=N0,seeds=SEEDS,delta=DELTA):
    xev=np.linspace(0,1,2001); fex=f(xev); den=np.linalg.norm(fex); rows=[]
    for name in NAMES:
        x,W,A=assemble_generic(name,kernel,N); U,s,Vt=np.linalg.svd(A,full_matrices=False); g=data_generic(kernel,f,x); vals=[]
        for seed in seeds:
            rng=np.random.default_rng(seed); e=rng.standard_normal(N); e=e/np.linalg.norm(e)*(delta*np.linalg.norm(g)); gd=g+e
            a,z,r=solve_morozov(U,s,Vt,gd,np.linalg.norm(e),tau=TAU); fr=reconstruct(name,N,W,z,xev)
            vals.append((a,np.linalg.norm(fr-fex)/den,np.max(np.abs(fr-fex)),r/np.linalg.norm(gd),(s[0]**2+a)/(s[-1]**2+a)))
        a=np.asarray(vals); mu=a.mean(0); sd=a.std(0,ddof=1); ci=1.96*sd/np.sqrt(len(seeds))
        rows.append(dict(problem=label,basis=name,N=N,delta=delta,runs=len(seeds),alpha_mean=mu[0],E2_mean=mu[1],E2_ci95=ci[1],Einf_mean=mu[2],Einf_ci95=ci[2],Eres_mean=mu[3],kappa_A=s[0]/s[-1],kappa_reg_mean=mu[4]))
    return rows

# Smooth but strongly oscillatory Hilbert--Schmidt kernel.
def Kosc(x,y): return np.exp(-((x-y)/0.20)**2)*np.cos(8*np.pi*(x-y))
OSC=stats_generic(Kosc,F,'oscillatory Gaussian-modulated kernel')

# Weakly singular Hilbert--Schmidt kernel |x-y|^{-beta}, beta<1/2.
BETA=0.25
zj,wj=roots_jacobi(120,0.0,-BETA); tj=(zj+1)/2; wsing=(2**(BETA-1))*wj

def singular_row_integral(x, values_fn):
    out=0.0
    if x>0:
        yl=x*(1-tj); out += x**(1-BETA)*np.tensordot(wsing,values_fn(yl),axes=(0,0))
    if x<1:
        yr=x+(1-x)*tj; out += (1-x)**(1-BETA)*np.tensordot(wsing,values_fn(yr),axes=(0,0))
    return out

def assemble_weak(name,N=N0):
    x=(np.arange(N)+0.5)/N; Bq=bmat(name,QX,N); _,W=mass_whitener(Bq); Ac=np.zeros((N,N))
    for i,xx in enumerate(x): Ac[i,:]=singular_row_integral(xx,lambda y:bmat(name,y,N))
    return x,W,Ac@W

def data_weak(f,x): return np.array([singular_row_integral(xx,lambda y:f(y)) for xx in x])

def stats_weak(f=F,N=N0,seeds=SEEDS,delta=DELTA):
    xev=np.linspace(0,1,2001); fex=f(xev); den=np.linalg.norm(fex); rows=[]
    for name in NAMES:
        x,W,A=assemble_weak(name,N); U,s,Vt=np.linalg.svd(A,full_matrices=False); g=data_weak(f,x); vals=[]
        for seed in seeds:
            rng=np.random.default_rng(seed); e=rng.standard_normal(N); e=e/np.linalg.norm(e)*(delta*np.linalg.norm(g)); gd=g+e
            a,z,r=solve_morozov(U,s,Vt,gd,np.linalg.norm(e),tau=TAU); fr=reconstruct(name,N,W,z,xev)
            vals.append((a,np.linalg.norm(fr-fex)/den,np.max(np.abs(fr-fex)),r/np.linalg.norm(gd),(s[0]**2+a)/(s[-1]**2+a)))
        a=np.asarray(vals); mu=a.mean(0); sd=a.std(0,ddof=1); ci=1.96*sd/np.sqrt(len(seeds))
        rows.append(dict(problem='weakly singular |x-y|^{-1/4}',basis=name,N=N,delta=delta,runs=len(seeds),alpha_mean=mu[0],E2_mean=mu[1],E2_ci95=ci[1],Einf_mean=mu[2],Einf_ci95=ci[2],Eres_mean=mu[3],kappa_A=s[0]/s[-1],kappa_reg_mean=mu[4]))
    return rows
WEAK=stats_weak()

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
write_csv(OUT/'challenging_kernel_statistics.csv',OSC+WEAK)

# Approximation-space convergence study.  This isolates the discretization order
# from regularization semi-convergence: f is projected in L2 onto V_N and the
# observed rate is log2(E_N/E_2N).  The inverse tests below separately document
# Morozov-regularized convergence/noise behavior.
CONV=[]
xe,we=np.polynomial.legendre.leggauss(600); xe=(xe+1)/2; we=we/2; fex=F(xe); fden=math.sqrt(float(np.sum(we*fex*fex)))
for name in NAMES:
    prev=None
    for N in [8,16,32,64]:
        B=bmat(name,xe,N); M=B.T@(we[:,None]*B); rhs=B.T@(we*fex)
        # Symmetric solve with a negligible diagonal safeguard for nonorthogonal bases.
        c=np.linalg.solve(M+1e-14*np.eye(N),rhs); fp=B@c
        err=math.sqrt(float(np.sum(we*(fp-fex)**2)))/fden
        rate='' if prev is None else math.log(prev/err,2)
        CONV.append(dict(basis=name,N=N,Eproj_L2=err,observed_rate=rate))
        prev=err
write_csv(OUT/'observed_convergence_rates.csv',CONV)

# Alpha sensitivity around the Morozov value for one common perturbation.
SENS=[]; kernel=R1['GAUSSIAN'](0.1); xev=np.linspace(0,1,2001); fex=F(xev); factors=np.logspace(-2,2,25)
for name in NAMES:
    x,W,A=assemble_generic(name,kernel,N0); U,s,Vt=np.linalg.svd(A,full_matrices=False); g=data_generic(kernel,F,x); rng=np.random.default_rng(4242); e=rng.standard_normal(N0); e=e/np.linalg.norm(e)*(DELTA*np.linalg.norm(g)); gd=g+e; am,zm,rm=solve_morozov(U,s,Vt,gd,np.linalg.norm(e),tau=TAU); ug=U.T@gd
    for fac in factors:
        aa=am*fac; z=Vt.T@((s/(s*s+aa))*ug); fr=reconstruct(name,N0,W,z,xev); SENS.append(dict(basis=name,factor=fac,alpha=aa,alpha_morozov=am,E2=np.linalg.norm(fr-fex)/np.linalg.norm(fex),residual=np.linalg.norm(A@z-gd)/np.linalg.norm(gd)))
write_csv(OUT/'alpha_sensitivity.csv',SENS)
plt.figure(figsize=(7.2,4.8))
for name in NAMES:
    rr=[r for r in SENS if r['basis']==name]; plt.loglog([r['factor'] for r in rr],[r['E2'] for r in rr],marker='o',ms=3,label=name)
plt.axvline(1.0,ls='--',lw=1,label='Morozov scale'); plt.xlabel(r'$\alpha/\alpha_{\rm Mor}$'); plt.ylabel('Relative $L^2$ solution error'); plt.grid(True,which='both',alpha=.25); plt.legend(ncol=2); plt.tight_layout(); plt.savefig(FIG/'fig_alpha_sensitivity.png',dpi=220); plt.close()

# Repeated CPU times for the three principal inverse examples.
TIM=[]
def time_case(label,name,assembler,datafun,reps=10):
    at=[]; st=[]
    for _ in range(reps):
        t=time.perf_counter(); x,W,A=assembler(name,N0); at.append(time.perf_counter()-t)
    g=datafun(F,x)
    for _ in range(50):
        t=time.perf_counter(); U,s,Vt=np.linalg.svd(A,full_matrices=False); aa=1e-3; _=Vt.T@((s/(s*s+aa))*(U.T@g)); st.append(time.perf_counter()-t)
    TIM.append(dict(problem=label,basis=name,assembly_median_s=float(np.median(at)),solve_median_s=float(np.median(st)),total_median_s=float(np.median(at)+np.median(st))))
for name in NAMES:
    time_case('Gaussian sigma=0.1',name,lambda nm,N:assemble_generic(nm,R1['GAUSSIAN'](0.1),N),lambda f,x:data_generic(R1['GAUSSIAN'](0.1),f,x))
    time_case('Oscillatory',name,lambda nm,N:assemble_generic(nm,Kosc,N),lambda f,x:data_generic(Kosc,f,x))
    time_case('Weakly singular',name,assemble_weak,lambda f,x:data_weak(f,x),reps=3)
write_csv(OUT/'cpu_times_all_inverse_examples.csv',TIM)

# Figures comparing challenging-kernel errors.
plt.figure(figsize=(7.2,4.8)); xp=np.arange(len(NAMES)); width=.36
osc=[next(r['E2_mean'] for r in OSC if r['basis']==n) for n in NAMES]; wk=[next(r['E2_mean'] for r in WEAK if r['basis']==n) for n in NAMES]
plt.bar(xp-width/2,osc,width,label='Oscillatory'); plt.bar(xp+width/2,wk,width,label='Weakly singular'); plt.yscale('log'); plt.xticks(xp,NAMES,rotation=20); plt.ylabel('Mean relative $L^2$ error'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'fig_challenging_kernels.png',dpi=220); plt.close()

print('Reviewer 2 additions written to',OUT)
