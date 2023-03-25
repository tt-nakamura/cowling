import numpy as np
import matplotlib.pyplot as plt
from cowling import CowlingModel
from constants import Msun,Rsun,Lsun,Tesun

c1 = CowlingModel(Z=0.02, X=0.7, X1=0.7)
c2 = CowlingModel(Z=0.02, X=0.7, X1=0.1, init=c1)

M = np.geomspace(Msun, 50*Msun, 100)
R1,L1,T1 = c1.RLT(M)
R2,L2,T2 = c2.RLT(M)

plt.figure(figsize=(5,6))

plt.subplot(311)
plt.plot(M/Msun, R1/Rsun, label=r'$X_1$=%g'%c1.X[0])
plt.plot(M/Msun, R2/Rsun, label=r'$X_1$=%g'%c2.X[0])
plt.ylabel(r'$R$  / $R_\odot$')
plt.legend()

plt.subplot(312)
plt.semilogy(M/Msun, L1/Lsun)
plt.semilogy(M/Msun, L2/Lsun)
plt.ylabel(r'$L$  / $L_\odot$')

plt.subplot(313)
plt.plot(M/Msun, T1/Tesun)
plt.plot(M/Msun, T2/Tesun)
plt.ylabel(r'$T_{\rm e}$  / $T_{\rm e\odot}$')
plt.xlabel(r'$M$ = mass  / $M_\odot$')

plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()
