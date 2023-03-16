#Algorithm for Path Weight Sampling in Langevin dynamics with feedback
#Output (standard output to screen console) is the accumulated Monte-Carlo average of MI and two transfer entropies, for every new trajectory
#Written by Avishek Das, Mar 16, 2023

import numpy as np
from numba import jit,njit

#Global variables
#2D Ornstein-Uhlenbeck process parameters
#where \dot{x}=-kxx*x-kxy*y+(2*Dx)**0.5 *white noise
#      \dot{y}=-kyx*x-kyy*y+(2*Dy)**0.5 *white noise
kxx=1
kxy=0.6
kyy=2
kyx=0.3
Dx=3
Dy=0.5
#timestep
dt=0.01
stepmags=np.array([10,100,1000]) #trajectory durations in increasing order, in number of timesteps for calculating total MI over
N=1000 #number of trajectories for outer expectation in MI expression
K=10000 #number of trajectories for each marginalization

####Given a 2D Langevin equation of noise magnitudes defined through Dx and Dy, for any system, only need to modify the four force functions below
@njit
def forcex(x,y): #force on x due to x and y
    return -kxx*x-kxy*y

@njit
def forcey(x,y): #force on y due to x and y
    return -kyx*x-kyy*y

@njit
def force0x(x): #force for a system where x evolves independently
    return -kxx*x

@njit
def force0y(y): #force for a system where y evolves independently
    return -kyy*y



####Other functions that need not be modified
@njit
def propagatexy(xi,yi,steps): #langevin equation to propagate x and y
    xt=np.zeros(steps+1)
    yt=np.zeros(steps+1)
    xt[0]=xi
    yt[0]=yi
    for i in range(0): #relaxation of initial condition #if this loop is entered, initial distribution must be accounted for in information calculation
        x=xt[0]
        y=yt[0]

        psi=np.random.normal()
        psi2=np.random.normal()
        xt[0]=x+forcex(x,y)*dt+(2*Dx*dt)**0.5*psi
        yt[0]=y+forcey(x,y)*dt+(2*Dy*dt)**0.5*psi2

    for i in range(steps):
        x=xt[i]
        y=yt[i]

        psi=np.random.normal()
        psi2=np.random.normal()
        xt[i+1]=x+forcex(x,y)*dt+(2*Dx*dt)**0.5*psi
        yt[i+1]=y+forcey(x,y)*dt+(2*Dy*dt)**0.5*psi2

    return xt,yt

@njit
def propagatey(yi,steps): #langevin equation for y in the absence of information transfer from x to y
    yt=np.zeros(steps+1)
    yt[0]=yi

    for i in range(0): #relaxation of initial condition
        y=yt[0]

        psi=np.random.normal()
        yt[0]=y+force0y(y)*dt+(2*Dy*dt)**0.5*psi

    for i in range(steps):
        y=yt[i]

        psi=np.random.normal()
        yt[i+1]=y+force0y(y)*dt+(2*Dy*dt)**0.5*psi

    return yt

@njit
def propagatex(xi,steps): #langevin equation for x in the absence of information transfer from y to x
    xt=np.zeros(steps+1)
    xt[0]=xi

    for i in range(0): #relaxation of initial condition
        x=xt[0]

        psi=np.random.normal()
        xt[0]=x+force0x(x)*dt+(2*Dx*dt)**0.5*psi

    for i in range(steps):
        x=xt[i]

        psi=np.random.normal()
        xt[i+1]=x+force0x(x)*dt+(2*Dx*dt)**0.5*psi

    return xt

#probabilities are represented in log scale all through
@njit
def trajprob(xt,yt,M,stepmags): #given two trajectories it returns log of joint probability P(X,Y), comprising of two terms useful for transfer entropy
    p=np.zeros((M,2))
    steps=int(stepmags[-1]) #maximum number of steps

    for i in range(steps):
        pstep1=(-(xt[i+1]-xt[i]-forcex(xt[i],yt[i])*dt)**2/4/Dx/dt)-np.log((4*np.pi*Dx*dt)**0.5) #Onsager Machlup actions minus normalization
        pstep2=(-(yt[i+1]-yt[i]-forcey(xt[i],yt[i])*dt)**2/4/Dy/dt)-np.log((4*np.pi*Dy*dt)**0.5)
        for j in range(M):
            if i<stepmags[j]:
                p[j,0]+=pstep1
                p[j,1]+=pstep2
    return p

@njit
def trajprob0x(xt,M,stepmags): #given xt it gives independent probability P_0(X)
    p=np.zeros(M)
    steps=int(stepmags[-1]) #maximum number of steps

    for i in range(steps):
        pstep=(-(xt[i+1]-xt[i]-force0x(xt[i])*dt)**2/4/Dx/dt)-np.log((4*np.pi*Dx*dt)**0.5)
        for j in range(M):
            if i<stepmags[j]:
                p[j]+=pstep
    return p

@njit
def trajprob0y(yt,M,stepmags): #given yt it gives independent probability P_0(Y)
    p=np.zeros(M)
    steps=int(stepmags[-1]) #maximum number of steps

    for i in range(steps):
        pstep=(-(yt[i+1]-yt[i]-force0y(yt[i])*dt)**2/4/Dy/dt)-np.log((4*np.pi*Dy*dt)**0.5)
        for j in range(M):
            if i<stepmags[j]:
                p[j]+=pstep
    return p

@njit
def marginalx(xt,M,stepmags,K): #given xt, compute marginal P(X) through Rosenbluth-Rosenbluth method
    steps=int(stepmags[-1]) #maximum number of steps
    ytrajs=np.zeros((K,steps+1))
    ytrajs[:,0]=0.0
    Uk=np.zeros(K)
    Ukt=Uk
    Umin=0.0
    wval=np.zeros(M) #marginal probability only upto the given duration
    Keff=K
    w=K
    Mcount=0

    for i in range(steps):
        #check if resampling is needed
        if Keff<K/2:
            for j in range(Mcount,M):
                wval[j]+=np.log(w/K)-Umin

            #resample with weights exponential of -Ukt
            ytrajs[:,i]=(ytrajs[:,i])[np.searchsorted(np.cumsum(np.exp(-Ukt)/w), np.random.random(), side="right")]

            #reset Uk
            Uk[:]=0.

        #propagate ensemble of y trajectories for one step and accumulate weight
        for k in range(K):
            ytrajs[k,i+1]=propagatey(ytrajs[k,i],1)[-1]
            p=trajprob(xt[i:i+2],ytrajs[k,i:i+2],1,np.array([1]))
            Uj=-np.sum(p[-1,:])
            U0=-trajprob0y(ytrajs[k,i:i+2],1,np.array([1]))[-1]
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk) 
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Mcount=Mcount+1

    return wval

@njit
def marginaly(yt,M,stepmags,K): #given yt, compute marginal P(Y) through Rosenbluth-Rosenbluth method
    steps=int(stepmags[-1]) #maximum number of steps
    xtrajs=np.zeros((K,steps+1))
    xtrajs[:,0]=0.0
    Uk=np.zeros(K)
    Ukt=Uk
    Umin=0.0
    wval=np.zeros(M) #marginal probability only upto the given duration
    Keff=K
    w=K
    Mcount=0

    for i in range(steps):
        #check if resampling is needed
        if Keff<K/2:
            for j in range(Mcount,M):
                wval[j]+=np.log(w/K)-Umin

            #resample with weights exponential of -Ukt
            xtrajs[:,i]=(xtrajs[:,i])[np.searchsorted(np.cumsum(np.exp(-Ukt)/w), np.random.random(), side="right")]

            #reset Uk
            Uk[:]=0.

        #propagate ensemble of x trajectories for one step and accumulate weight
        for k in range(K):
            xtrajs[k,i+1]=propagatex(xtrajs[k,i],1)[-1]
            p=trajprob(xtrajs[k,i:i+2],yt[i:i+2],1,np.ones(1))
            Uj=-np.sum(p[-1,:])
            U0=-trajprob0x(xtrajs[k,i:i+2],1,np.ones(1))[-1]
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Mcount=Mcount+1

    return wval

@njit
def rrpws():
    M=np.shape(stepmags)[0]

    Iest=np.zeros(M)
    Test=np.zeros((M,2))

    xts=np.zeros((N,stepmags[-1]+1))
    yts=np.zeros((N,stepmags[-1]+1))

    np.random.seed(2) #change the seed to change the statistical realizations, or eliminate this altogether for a random seed

    for i in range(N):
        xts[i,:],yts[i,:]=propagatexy(0.,0.,stepmags[-1]) # 0 and 0 are initial values for starting x and y trajectories

    for i in range(N):
        xt=xts[i,:]
        yt=yts[i,:]

        p=trajprob(xt,yt,M,stepmags)
        Iest+=np.sum(p,axis=-1) #numerator in MI expression
        Test+=p #numerators in transfer entropy expression

        px=marginalx(xt,M,stepmags,K) #compute with RR-PWS the marginal probability of xt
        py=marginaly(yt,M,stepmags,K) #compute with RR-PWS the marginal probability of yt
        
        Iest-=px+py #MI denominator
        Test[:,0]-=px #transfer entropy denominator
        Test[:,1]-=py #transfer entropy denominator

        #with numba.objmode():
        print(Iest[0]/(i+1),Iest[1]/(i+1),Iest[2]/(i+1))#,flush=True)
        print(Test[0,0]/(i+1),Test[1,0]/(i+1),Test[2,0]/(i+1))
        print(Test[0,1]/(i+1),Test[1,1]/(i+1),Test[2,1]/(i+1))

    return

rrpws()

