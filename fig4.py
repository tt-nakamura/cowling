import numpy as np
import matplotlib.pyplot as plt
from cowling import CowlingModel
from constants import Msun,Lsun,Tesun
from scipy.constants import year

Myr = 1e6*year

m = 9
M = np.linspace(2*Msun, 10*Msun, m)
X,Z = 0.7, 0.02
dt,n,nstep = 0.01*Myr, 2, 20

L,T = np.empty((2,m,n+1))
t = np.arange(n+1)*dt*nstep/Myr

for i in range(m):
    X1 = X
    c = CowlingModel(Z,X,X1)
    c.set_mass(M[i])
    L[i,0],T[i,0] = c.L,c.T
    for j in range(1,n+1):
        for _ in range(nstep):
            c = CowlingModel(Z,X,X1,init=c)
            c.set_mass(M[i])
            X1 += dt*c.dX_dt()

        L[i,j],T[i,j] = c.L,c.T

plt.figure(figsize=(5,3.75))

for j in range(n+1):
    plt.loglog(T[:,j]/Tesun, L[:,j]/Lsun, '+:',
               label=r'$t$=%g Myr'%t[j])

plt.xlim([11,2])
plt.ylabel(r'$L$ = luminosity  / $L_\odot$')
plt.xlabel(r'$T_{\rm e}$ = surface temperature  / $T_{\rm e\odot}$')
plt.legend()
plt.tight_layout()
plt.savefig('fig4.eps')
plt.show()
