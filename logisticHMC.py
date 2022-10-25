# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 09:10:29 2021

@author: shaw
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from scipy.linalg import cholesky,cho_solve, solve_triangular
from scipy.optimize import fsolve,minimize
from scipy.special import expit
from emcee.autocorr import integrated_time
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

import matplotlib.pyplot as plt
import pickle
plt.rc('text', usetex=True)
plt.rc('font',**{'serif':['cm']})
plt.style.use('seaborn-paper')
plt.rcParams.update({'axes.titlesize':50,'axes.labelsize': 40,'xtick.labelsize': 30,'ytick.labelsize': 30,
                    'legend.fontsize': 30,'legend.title_fontsize':35,'figure.titlesize': 60,'lines.linewidth' : 2})
import time as time
def rotate(qp,v,h):
    q_=np.cos(h)*qp;v_=np.cos(h)*v
    q_+=v*np.sin(h);v_-=np.sin(h)*qp
    return q_,v_        

plotdict={'UncondVerlet':'s','UncondKRK':'X','PrecondVerlet':'d','PrecondRKR':'o','PrecondKRK':'*'}
#####Calculate ACT#######

def calc_ll(experiment,q):
    arg=q@(experiment.xnew.T)
    ans=np.sum(experiment.y*arg,axis=1)
    ans-=np.sum(np.logaddexp(0.0,arg),axis=1)
    return ans.flatten()

def calc_beta(q):
    squares=np.sum(q**2,axis=1)
    return squares.flatten()

def calc_taubatch(q):
    '''
    Method of batch means
    '''
    S2=np.var(q,axis=0)
    Ns=q.shape[0] #number of samples
    bs=int(Ns**(2/3))
    batches=int(Ns**(1/3))
    my_means=q[:batches*bs].reshape(batches,bs,-1)
    batch_means=np.mean(my_means,axis=1)
    Sb2=np.var(batch_means,axis=0)
    act=bs*Sb2/S2
    return act.flatten()

def calc_slowcomp(q):
    act=-1
    j=0
    for i in range(q.shape[-1]):
        actnew=integrated_time(q[:,i])
        if actnew>act: 
            act=actnew
            j=i
    return act,j


class HMCIntegrators:
    
    def __init__(self, J, Jchol, Nsamples, MAP, Z, freqs):
        self.J=J
        self.Jchol=Jchol
        self.Nsamples=Nsamples
        self.MAP=MAP
        self.Z=Z
        self.Z_T=Z.T
        self.freqs=freqs
    
    def U(self,q):
        return 
    
    def grad(self,q):
        return 
    
    def partialgrad(self,q):
        U_dash=self.grad(q)
        return U_dash-self.J@(q-self.MAP)
        
    def UncondRotate(self,q,p,h):
        hw=self.freqs*h
        q_=self.Z@q;p_=self.Z@p
        qnew=np.cos(hw)*q_+np.sin(hw)*p_/self.freqs
        pnew=np.cos(hw)*p_-np.sin(hw)*q_*self.freqs
        return self.Z_T@qnew,self.Z_T@pnew
    
    def Unconditioned(self,q0,hmed,T,integrat='KRK'):
        switcher = {'KRK':self.UncondKRK,'Verlet':self.UncondVerlet,
                    'RKR':self.UncondRKR}
        if integrat not in switcher.keys():
            raise ValueError(f"integrat argument ({integrat}) of Unconditioned arg is not valid."+
                             " Only 'KRK','RKR' or 'Verlet' are acceptable.")
        q=q0.copy()
        D=len(q0)
        acc=0
        ham=lambda q,p:.5*np.dot(p,p)+self.U(q)
        Nsamples=self.Nsamples
        samples=np.zeros((Nsamples,D))
        Nsteps=int(np.floor(T/hmed))
        integrator=switcher.get(integrat)
        for n in range(0,Nsamples):
            h=np.random.uniform(low=.8,high=1.0)*hmed
            p=np.random.normal(size=D) #Draw p ~ N(0,I)
            qp=q.copy()
            H0=ham(q,p)
            
            #Do a leg of T//h steps of Strang
            qp,p=integrator(qp,p,Nsteps,h)
            
            accept=H0-ham(qp,p) #acceptance probability
            #Accept/reject
            if (accept>np.log(np.random.uniform())):
                acc+=1
                q=qp 
            else:
                pass
            samples[n]=q
        return acc/Nsamples,samples
    
    def Preconditioned(self,q0,hmed,T,integrat='KRK'):

        switcher = {'KRK':self.PrecondKRK,'Verlet':self.PrecondVerlet,
                    'RKR':self.PrecondRKR}
        if integrat not in switcher.keys():
            raise ValueError(f"integrat argument ({integrat}) of Preconditioned arg is not valid."+
                             " Only 'KRK','RKR' or 'Verlet' are acceptable.")
        
        q=q0.copy()
        D=len(q0)
        acc=0
        Nsamples=self.Nsamples
        ham=lambda q,v:.5*np.dot(v,self.J@v)+self.U(q)
        samples=np.zeros((Nsamples,D))
        Nsteps=int(np.floor(T/hmed))
        integrator=switcher.get(integrat)
        for n in range(0,Nsamples):
            h=np.random.uniform(low=.8,high=1.0)*hmed
            v=solve_triangular(self.Jchol, np.random.normal(size=D),lower=False) #Draw v ~ N(0,Jinv)
            qp=q.copy()
            H0=ham(q,v)
            
            #Do a leg of T//h steps of Strang
            qp,v=integrator(qp,v,Nsteps,h)

            accept=H0-ham(qp,v) #acceptance probability
            #Accept/reject
            if (accept>np.log(np.random.uniform())):
                acc+=1
                q=qp 
            else:
                pass
            samples[n]=q
        return acc/Nsamples,samples

    ##UncondIntegrators##
    def UncondVerlet(self,qp,p,Nsteps,h):
        #Do a leg of T//h steps of Strang
        #(b1) Kick
        theta1=h/2
        p-=theta1*self.grad(qp)
        for t in np.arange(0,Nsteps):
            qp+=h*p #Drift
            theta = 2*theta1 if (t!=Nsteps-1) else theta1
            p-=theta*self.grad(qp)
        return qp,p
    def UncondKRK(self,qp,p,Nsteps,h):
        #Do a leg of T//h steps of Strang
        #(b1) Kick
        theta1=h/2
        p-=theta1*self.partialgrad(qp)
        for t in np.arange(0,Nsteps):
            qp-=self.MAP
            qp,p=self.UncondRotate(qp,p,h) #Rotate
            qp+=self.MAP
            theta = 2*theta1 if (t!=Nsteps-1) else theta1
            p-=theta*self.partialgrad(qp)
        return qp,p
    def UncondRKR(self,qp,p,Nsteps,h):
        #Do a leg of T//h steps of Strang
        #(b1) Kick
        theta1=h/2
        qp-=self.MAP
        qp,p=self.UncondRotate(qp,p,theta1) #Rotate
        qp+=self.MAP
        for t in np.arange(0,Nsteps):
            p-=h*self.partialgrad(qp)
            theta = 2*theta1 if (t!=Nsteps-1) else theta1
            qp-=self.MAP
            qp,p=self.UncondRotate(qp,p,theta) #Rotate
            qp+=self.MAP        
        return qp,p
    
    ##PrecondIntegrators##
    def PrecondVerlet(self,qp,v,Nsteps,h):
        #Do a leg of T//h steps of Strang
        #(b1) Kick
        theta1=h/2
        v-=theta1*cho_solve((self.Jchol,False), self.grad(qp))
        for t in np.arange(0,Nsteps):
            qp+=h*v #Drift
            theta = 2*theta1 if (t!=Nsteps-1) else theta1
            v-=theta*cho_solve((self.Jchol,False), self.grad(qp))
        return qp,v
    def PrecondKRK(self,qp,v,Nsteps,h):
        #Do a leg of T//h steps of Strang
        #(b1) Kick
        theta1=h/2
        v-=theta1*cho_solve((self.Jchol,False), self.partialgrad(qp))
        for t in np.arange(0,Nsteps):
            qp-=self.MAP
            qp,v=rotate(qp,v,h) #Rotate
            qp+=self.MAP
            theta = 2*theta1 if (t!=Nsteps-1) else theta1
            v-=theta*cho_solve((self.Jchol,False), self.partialgrad(qp))
        return qp,v
    def PrecondRKR(self,qp,v,Nsteps,h):
        #Do a leg of T//h steps of Strang
        #(b1) Rotate
        theta1=h/2
        qp-=self.MAP
        qp,v=rotate(qp,v,theta1)
        qp+=self.MAP
        for t in np.arange(0,Nsteps):
            v-=h*cho_solve((self.Jchol,False), self.partialgrad(qp)) #kick
            theta = 2*theta1 if (t!=Nsteps-1) else theta1
            qp-=self.MAP
            qp,v=rotate(qp,v,theta)
            qp+=self.MAP
        return qp,v


class LogRegExp(HMCIntegrators):
    def __init__(self,Nsamples,C,x,y,poly_order=1):
        self.C=C
        self.x=x
        self.y=y
        self.n=x.shape[0] # # of data points, which should always be > # of params
        if n<np.max(x.shape):
            raise ValueError('x is likely of wrong shape. Try taking the Transpose.')
        self.xnew=PolynomialFeatures(poly_order,interaction_only=True).fit_transform(x)
        self.num_params=np.shape(self.xnew)[1]
        MAP=self.calc_MAP()
        
        #calculate hessian at MAP
        arg=self.xnew@MAP
        J=self.xnew.T*(expit(arg)*expit(-arg))@self.xnew
        np.fill_diagonal(J, J.diagonal() + 1/self.C)
        Jchol=cholesky(J, lower=False)
        eigs, mat = np.linalg.eigh(J)
        if np.min(eigs)<0: raise Exception("J found as Hessian of U at MAP estimate q_hat is not positive-semidefinite!")
        Z=mat.T #J is symm pos-def and so diagonalisable by orthogonal matrices
        freqs=np.sqrt(eigs)
        super().__init__(J, Jchol, Nsamples, MAP,Z,freqs)

    def U(self,q):
        arg=self.xnew@q
        ans=-np.dot(self.y,arg)
        ans+=np.sum(np.logaddexp(0.0,arg))
        return .5*(np.dot(q,q/self.C))+ans
    
    def grad(self,q):
        return q/self.C-self.xnew.T@(self.y-expit(self.xnew@q))
    
    def calc_MAP(self):
        x0=np.random.normal(size=self.num_params,scale=.2)
        guess=fsolve(self.grad,x0=x0)
        return minimize(self.U,x0=guess).x


#Sim Data
#Generate data for Logistic Regression
np.random.seed(2022)
d=100
p=d+1
n=20000
scaler=np.hstack((5*np.ones(shape=(1,5)),np.ones(shape=(1,5)),.2*np.ones(shape=(1,d-10))))
params=np.array([-0.65480083,-0.02877456,-0.19413575,-0.90141523,1.31329723,-0.82243619
,-0.25875645,0.23465318,-0.42060734,-0.63676356,0.18619602,0.67633567
,-0.33292241,0.34700941,-1.31302660,0.17623538,0.22350838,0.89700385
,1.26243062,0.85263681,-0.18392176,-0.71155761,0.53023805,0.22013331
,0.77031093,-0.97984402,1.52837540,-0.53118770,-0.29079245,-0.81254984
,-0.26238828,1.24624430,0.85436712,0.02894546,0.61802959,-0.03663088
,0.28129619,-0.46559747,0.05948328,-0.03556713,-0.18918631,0.32643383
,0.84860412,-0.54256356,-0.81374244,-1.53385878,0.30440185,1.23163473
,-1.27378357,-1.06841908,-0.03488537,0.71541162,-0.82484065,1.74136932
,1.92796733,0.05423268,-0.91609818,0.53988461,-1.24644228,-2.57801148
,0.20859099,-2.34587525,-0.53159755,-0.23084366,-2.18340432,-0.17718191
,0.03045268,0.40114077,0.36879257,-0.66655558,0.65313015,-1.56958581
,-0.77240783,1.56132134,-1.12741534,1.35426177,-0.02357929,0.89622097
,1.29100981,-0.08017747,1.25847627,0.32861204,-0.97089441,0.59948519
,-0.18790193,1.34199773,-0.87842449,1.02578706,-0.03887998,-2.35760921
,-2.07777295,-0.61723568,-0.34878943,0.61748140,-0.24616670,-2.84947421
,1.47901785,1.62453950,-1.97244995,-0.64548469,0.28893605])
xdata=np.random.normal(size=(n,d),scale=scaler) #input data
xnew=np.hstack((np.ones(shape=(n,1)),xdata))
p_i=expit(xnew@params)
ydata=np.random.binomial(1, p_i).flatten() # output data
sig_sq=25 #priors
C=sig_sq

#Nrange Data plot
Nsamples=5000
nrange=np.int64(2**(np.arange(7,np.log2(n),.5)))
Lp=2
LpV=3
LuV=Lu=30
Tp=Tu=np.pi/2
samples={'UncondVerlet':{'acc':[],'T':[],'h':[]},'UncondKRK':{'acc':[],'T':[],'h':[]},
          'PrecondVerlet':{'acc':[],'T':[],'h':[]},'PrecondKRK':{'acc':[],'T':[],'h':[]},'PrecondRKR':{'acc':[],'T':[],'h':[]}
          'L_UVerlet':LuV, 'L_Uncond':Lu, 'L_Precond':Lp,'L_PVerlet':LpV,'T_U':Tu,'T_P':Tp,'scale_by_freq':True,'nrange':nrange}

for n in nrange:
    x=xdata[:n]#input data
    y=ydata[:n] # output data
    Exp1=LogRegExp(Nsamples, C, x, y)
    min_freq=np.min(Exp1.freqs)
    q0=Exp1.MAP
    Tu_=Tu/min_freq
    for k,TL in {'UncondVerlet':(Tu_,LuV),'UncondKRK':(Tu_,Lu),'PrecondVerlet':(Tp,LpV),
                  'PrecondKRK':(Tp,Lp),'PrecondRKR':(Tp,Lp)}.items():
        integrator=k[-3:]
        T,L=TL
        h_=T/L
        if k[-1]=='t': integrator='Verlet'
        if k[0]=='U':
            acc,s=Exp1.Unconditioned(q0, h_, Tu,integrat=integrator)
        else:
            acc,s=Exp1.Preconditioned(q0, h_, Tp,integrat=integrator)
        samples[k]['acc']+=[acc]
        samples[k]['T']+=[T];samples[k]['h']+=[h_]
         

fig=plt.figure(figsize=(20,10))
for k,i in plotdict.items():
    label=k
    plt.semilogx(samples['nrange'],samples[k]['acc'],color='k',ls='-',label=label,ms=15,marker=i,base=2)
plt.xlabel('$n$')
plt.ylabel('AP')
plt.legend()
plt.title('SimData: Acceptance Rate as function of $n$')
fig.savefig('SimDatanrange.pdf',format='pdf',bbox_inches='tight')
exp1_file = open("SimDatanrange.pkl", 'wb')
pickle.dump(samples,exp1_file)
exp1_file.close()   
############

#Spectrum Plot
nrange=np.int64(2**(np.arange(7,20,2)))
my_freqs=np.zeros((len(nrange),p))
exact=np.zeros((p,p))
for i,n in enumerate(nrange):
    for it in range(5):
        x=scaler*np.random.normal(size=(n,d)) #input data
        xnew=np.hstack((np.ones(shape=(n,1)),x))
        p_i=expit(xnew@params)
        y=np.random.binomial(1, p_i).flatten() # output data
        Exp1=LogRegExp(1, C, x, y)
        my_freqs[i]+=np.flip(np.sort(Exp1.freqs))/5
        if i==len(nrange)-1: 
            p_i=np.reshape(p_i,newshape=(-1,1))
            exact+=xnew.T@(xnew*(p_i*(1-p_i)))/(5*n)

plotdata={'my_freqs':my_freqs,'p':p,'nrange':nrange}
eigs=np.linalg.eigvals(exact)
exact=np.flip(np.sort(np.sqrt(eigs)))
plotdata['exact']=exact

exp1_file = open("SimDataSpectrum.pkl", 'wb')
pickle.dump(plotdata,exp1_file)
exp1_file.close() 

# p=plotdata['p']
# nrange=plotdata['nrange']
# my_freqs=plotdata['my_freqs']

exact=plotdata['exact']
fig=plt.figure(figsize=(20,10))
for i,m in zip(range(len(nrange[:5])),['s','X','d','o','*']):
    plottable=my_freqs[i]/np.sqrt(plotdata['nrange'][i])
    plt.plot(np.arange(1,16),plottable[:15],color='k',ls=':',label='$n=2^{'+f'{np.log2(nrange[i]):.0f}'+'}$',ms=15,marker=m)
plt.plot(np.arange(1,16),exact[:15],color='k',ls='-',alpha=1,label='$\mathcal{I}_F$',lw=4)
plt.xlabel('Component $j$')
plt.ylabel('$\\omega_j/\sqrt{n}$')
plt.legend()
plt.title('SimData: Spectrum of $\mathcal{J}$'+f', \n $15$ largest eigenvalues for $d={p}$')
fig.savefig('SimDataSpectrum.pdf',format='pdf',bbox_inches='tight')


###############################################################################
####Reproduce Shahbaba Results and Tables#######
Nsamples=50000
sig_sq=25 #priors

def doExperiment(Exp1,huV,hu,hp,hpV,Tu,Tp,q0):
    Nsamples=Exp1.Nsamples
    print(f'min freq={np.min(Exp1.freqs):.2f}. max freq={np.max(Exp1.freqs):.2f}')

    samples={'Results':{},'Exp':Exp1}
    ints=['Verlet','KRK','RKR']
    switcher = {'Verlet':huV}
    for integrator in ints:
        #Unconditioned
        h_=switcher.get(integrator,hu) #default to hu if integrator!=Verlet
        start=time.time()
        acc,s=Exp1.Unconditioned(q0, h_, Tu,integrat=integrator)
        end=time.time()
        k='Uncond'+integrator
        samples['Results'][k]={'acc':acc,'samples':s,'h':h_,'T':Tu}
        samples['Results'][k]['exec_time']=end-start
        print(k+f', acc={acc}')

        #Preconditioned
        h_=hpV if integrator=='Verlet' else hp
        start=time.time()
        acc,s=Exp1.Preconditioned(q0, h_, Tp,integrat=integrator)
        end=time.time()
        k='Precond'+integrator
        samples['Results'][k]={'acc':acc,'samples':s,'h':h_,'T':Tp}
        samples['Results'][k]['exec_time']=end-start
        print(k+f', acc={acc}')
    
    for method in samples['Results'].keys():
        samples['Results'][method]['act']={}
        s=samples['Results'][method]['samples']
        ll=calc_ll(Exp1,s)
        beta=calc_beta(s)
        for name,func in zip(['batch','emcee'],[calc_taubatch,integrated_time]):
            act=func(ll)[0]
            samples['Results'][method]['act']['ll_'+name]=act
            act=func(beta)[0]
            samples['Results'][method]['act']['beta_'+name]=act
        samples['Results'][method]['act']['slowcomp_batch']=np.max(calc_taubatch(s))
        act,j=calc_slowcomp(s)
        samples['Results'][method]['act']['slowcomp_emcee']=act[0]
        plot_acf(s[:,j], title=method+' act=' +str(round(act[0],2)))

    for l in samples['Results'].keys():
        thing=samples['Results'][l]
        print(l+f"& & {thing['exec_time']/Nsamples:.2e} & {thing['act']['ll_emcee']:.1f} & {thing['act']['beta_emcee']:.1f} &{thing['act']['slowcomp_emcee']:.1f}& {thing['acc']:.2f}\ \ \hline")
    return samples


def doShahbaba(Exp1,huV,hu,Tu,q0):
    Nsamples=Exp1.Nsamples
    p=np.min((Exp1.x).shape)+1

    samples={'Results':{'UncondVerlet':{'acc':0,'samples':np.zeros((Nsamples,p)),'h':huV,'T':Tu},
             'UncondKRK':{'acc':0,'samples':np.zeros((Nsamples,p)),'h':hu,'T':Tu}},'Exp':Exp1}
    ints=['Verlet','KRK']
    for integrator in ints:
        #Unconditioned
        h_=huV if integrator=='Verlet' else hu 
        start=time.time()
        acc,s=Exp1.Unconditioned(q0, h_, Tu,integrat=integrator)
        end=time.time()
        k='Uncond'+integrator
        samples['Results'][k]['acc']=acc;samples['Results'][k]['samples']=s;samples['Results'][k]['exec_time']=end-start
        print(k+f', acc={acc}')

    for method in samples['Results'].keys():
        samples['Results'][method]['act']={}
        s=samples['Results'][method]['samples']
        ll=calc_ll(Exp1,s)
        beta=calc_beta(s)
        for name,func in zip(['batch','emcee'],[calc_taubatch,integrated_time]):
            act=func(ll)[0]
            samples['Results'][method]['act']['ll_'+name]=act
            act=func(beta)[0]
            samples['Results'][method]['act']['beta_'+name]=act
        samples['Results'][method]['act']['slowcomp_batch']=np.max(calc_taubatch(s))
        act,j=calc_slowcomp(s)
        samples['Results'][method]['act']['slowcomp_emcee']=act[0]
        plot_acf(s[:,j], title=method+' act=' +str(round(act[0],2)))
    return samples

#############
##Experiment 1: Sim Data
np.random.seed(2022)
d=100
p=d+1
n=10000
scaler=np.hstack((5*np.ones(shape=(1,5)),np.ones(shape=(1,5)),.2*np.ones(shape=(1,d-10))))
x=np.random.normal(size=(n,d),scale=scaler) #input data
xnew=np.hstack((np.ones(shape=(n,1)),x))
p_i=expit(xnew@params)
y=np.random.binomial(1, p_i).flatten() # output data

C=sig_sq
Exp1=LogRegExp(Nsamples, C, x, y,poly_order=1)
q0=Exp1.MAP
min_freq=np.min(Exp1.freqs)

#Principled parameter choice
Tu=np.pi/(2*min_freq)
hV=Tu/40
hS=Tu/20
Tp=np.pi/2
hP=Tp
hpV=Tp/3
samples=doExperiment(Exp1,hV,hS,hP,hpV,Tu,Tp,q0)

exp1_file = open("LogReg_SimData.pkl", 'wb')
pickle.dump(samples,exp1_file)
exp1_file.close()

#Shahbaba parameter choice
hV=.015
Tu=.3
hS=.03
samples=doShahbaba(Exp1,hV,hS,Tu,q0)

exp1_file = open("LogReg_SimDataShahbabaParams.pkl", 'wb')
pickle.dump(samples,exp1_file)
exp1_file.close()
#############

#############
####Experiment 2: Chess###
np.random.seed(2022)
data = pd.read_table('chess.txt', sep=",", header=None)
y = np.array(data.iloc[:,-1]=='won',dtype=np.float64)
X = data.iloc[:,:-1]
x = np.zeros_like(X,dtype=np.float64)
for i in range(x.shape[-1]): 
    x[:,i] = pd.factorize(X.iloc[:,i],sort=True)[0]

d=np.shape(x)[-1]
C=sig_sq
Exp1=LogRegExp(Nsamples, C, x, y)
q0=Exp1.MAP
min_freq=np.min(Exp1.freqs)

#Principled Parameter Choice
hV=.087
Tu=np.pi/(2*min_freq)
hS=.142
Tp=np.pi/2
hP=Tp/2
hpV=Tp/2

samples=doExperiment(Exp1,hV,hS,hP,hpV,Tu,Tp,q0)

exp1_file = open("LogReg_Chess.pkl", 'wb')
pickle.dump(samples,exp1_file)
exp1_file.close()

#Shahbaba parameter choice
hV=.09
Tu=20*.09
hS=.2
samples=doShahbaba(Exp1,hV,hS,Tu,q0)

exp1_file = open("LogReg_ChessShahbabaParams.pkl", 'wb')
pickle.dump(samples,exp1_file)
exp1_file.close()
#############

#############
####Experiment 3: CTG###
np.random.seed(2022)
ctg = pd.read_table('CTG.txt',header=0)
X = np.array(ctg.iloc[:,:21])
x = StandardScaler().fit_transform(X)
n=x.shape[0]
y = np.array(ctg.iloc[:,-1])
y=np.where(y>2,1,0)

d=np.shape(x)[-1]
C=sig_sq
Exp1=LogRegExp(Nsamples, C, x, y,poly_order=1)
q0=Exp1.MAP
min_freq=np.min(Exp1.freqs)

#Principled Parameter Choice
hV=.08
Tu=np.pi/(2*min_freq)
hS=.118

Tp=np.pi/2
hP=Tp/2
hpV=Tp/2
samples=doExperiment(Exp1,hV,hS,hP,hpV,Tu,Tp,q0)

exp1_file = open("LogReg_CTG.pkl", 'wb')
pickle.dump(samples,exp1_file)
exp1_file.close()

#Shahbaba parameter choice
hV=.08
Tu=20*.08
hS=Tu/13
samples=doShahbaba(Exp1,hV,hS,Tu,q0)

exp1_file = open("LogReg_CTGShahbabaParams.pkl", 'wb')
pickle.dump(samples,exp1_file)
exp1_file.close()
#############

#############
####Experiment 4: StatLog###
np.random.seed(2022)
data = pd.read_table('satTrn.txt', header=None, sep=' ')
X = np.array(data.iloc[:,:-1])
x = StandardScaler().fit_transform(X)
n=x.shape[0]
y = np.array(data.iloc[:,-1])
y=np.where(y==2,1,0)

d=np.shape(x)[-1]
C=sig_sq
Exp1=LogRegExp(Nsamples, C, x, y)
q0=Exp1.MAP
min_freq=np.min(Exp1.freqs)

#Principled Parameter Choice
hV=.08
Tu=np.pi/(2*min_freq)
hS=.114

Tp=np.pi/2
hP=Tp/2
hpV=Tp/3

samples=doExperiment(Exp1,hV,hS,hP,hpV,Tu,Tp,q0)

exp1_file = open("LogReg_StatLog.pkl", 'wb')
pickle.dump(samples,exp1_file)
exp1_file.close()

#Shahbaba parameter choice
hV=.08
Tu=20*.08
hS=Tu/14
samples=doShahbaba(Exp1,hV,hS,Tu,q0)

exp1_file = open("LogReg_StatLogShahbabaParams.pkl", 'wb')
pickle.dump(samples,exp1_file)
exp1_file.close()
#############


#############
exps=['SimData','StatLog','CTG','Chess']
# ###Autocorr
for k in exps:
    exp1_file = open("LogReg_"+k+".pkl", 'rb')
    samples=pickle.load(exp1_file)
    exp1_file.close()
    exp1_file = open("LogReg_"+k+"ShahbabaParams.pkl", 'rb')
    samples2=pickle.load(exp1_file)
    exp1_file.close()
    fig=plt.figure(figsize=(20,10))
    for l,m in plotdict.items():
        if l[0]=='U':
            name=l+'A'
            s=samples2['Results'][l]['samples']
            act,j=calc_slowcomp(s)
            data=acf(s[:,j])
            plt.plot(data,'k--',ms=15,marker=m,label=name,markevery=5)
            
        s=samples['Results'][l]['samples']
        name=l 
        if l[0]=='U': name=l+'B'
        act,j=calc_slowcomp(s)
        data=acf(s[:,j])
        plt.plot(data,'k-', ms=15,marker=m,label=name,markevery=5)
        
    plt.legend(ncol=2)
    plt.xlabel('Lag');plt.ylabel('Autocorrelation')
    fig.suptitle(k)
    fig.savefig(k+'Autocorr.pdf',format='pdf',bbox_inches='tight')

#Print tables
for k in exps:
    print(k)
    print("& g & $s$ [ms] & $\\tau_{\\ell}\\times s$  & $\\tau_{\\theta^2}\\times s$ &$\\tau_{\max}\\times s$ & AP \ \ \hline\hline")
    exp1_file = open("LogReg_"+k+".pkl", 'rb')
    samples=pickle.load(exp1_file)
    exp1_file.close()
    exp1_file = open("LogReg_"+k+"ShahbabaParams.pkl", 'rb')
    samples2=pickle.load(exp1_file)
    exp1_file.close()
    Nsamples=samples['Exp'].Nsamples
    for l in plotdict.keys():
        if l[0]=='U':
            thing=samples2['Results'][l]
            etime=(thing['exec_time']/Nsamples)*10**3
            g=int(thing['T']/thing['h']+.05)+1
            ll=round(thing['act']['ll_emcee'],1)
            beta=round(thing['act']['beta_emcee'],1)
            sc=round(thing['act']['slowcomp_emcee'],1)
            print(l+' A'+f"& {g} & {etime:.2f} & ${ll}\\times s={ll*etime:.1f}$ & ${beta}\\times s={beta*etime:.1f}$ & ${sc}\\times s={sc*etime:.1f}$  & {thing['acc']:.2f}\ \ \hline")
        thing=samples['Results'][l]
        name=l 
        etime=(thing['exec_time']/Nsamples)*10**3
        if l[0]=='U': name=l+' B'
        g=int(thing['T']/thing['h']+.05)+1*(l[-1]=='K' or l[-1]=='t')
        ll=round(thing['act']['ll_emcee'],1)
        beta=round(thing['act']['beta_emcee'],1)
        sc=round(thing['act']['slowcomp_emcee'],1)
        print(name+f"& {g} & {etime:.2f} & ${ll}\\times s={ll*etime:.1f}$ & ${beta}\\times s={beta*etime:.1f}$ & ${sc}\\times s={sc*etime:.1f}$ & {thing['acc']:.2f}\ \ \hline")
