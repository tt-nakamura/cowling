import numpy as np
import matplotlib.pyplot as plt
from cowling import CowlingModel
from constants import Msun,Rsun,Lsun,Tesun
from scipy.constants import year

Myr = 1e6*year

M,X,Z = 2*Msun, 0.7, 0.02
dt,n = 3*Myr, 113

c,X1 = None,X
R,L,T,Xc = np.empty((4,n))
t = np.arange(n)*dt

for i in range(n):
    c = CowlingModel(Z,X,X1,init=c)
    c.set_mass(M)
    X1 += dt*c.dX_dt()
    R[i],L[i],T[i],Xc[i] = c.R,c.L,c.T,X1

plt.figure(figsize=(5,8))

plt.subplot(411)
plt.plot(t/Myr, Xc)
plt.ylabel(r'$X_1$')

plt.subplot(412)
plt.plot(t/Myr, R/Rsun)
plt.ylabel(r'$R$  / $R_\odot$')

plt.subplot(413)
plt.plot(t/Myr, L/Lsun)
plt.ylabel(r'$L$  / $L_\odot$')

plt.subplot(414)
plt.plot(t/Myr, T/Tesun)
plt.ylabel(r'$T_{\rm e}$  / $T_{\rm e\odot}$')

plt.xlabel(r't = time  / Myr')

plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()
