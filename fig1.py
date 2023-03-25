import numpy as np
import matplotlib.pyplot as plt
from cowling import CowlingModel
from constants import Msun,Rsun,Tsun,Psun,Lsun

plt.figure(figsize=(5,8))

c = CowlingModel(Z=0.02, X=0.7, X1=0.1)
c.set_mass(2*Msun)
r,m,l,P,T,rho = c.rmlPTrho()
n = c.Nf # index of fitting point

plt.subplot(411)
plt.plot(r/Rsun, m/Msun)
plt.plot(r[n]/Rsun, m[n]/Msun, 'r.')
plt.ylabel(r'$m$ = mass  / $M_\odot$')

plt.subplot(412)
plt.plot(r/Rsun, l/Lsun)
plt.plot(r[n]/Rsun, l[n]/Lsun, 'r.')
plt.ylabel(r'$l$ = luminosity  / $L_\odot$')

plt.subplot(413)
plt.plot(r/Rsun, P/Psun)
plt.plot(r[n]/Rsun, P[n]/Psun, 'r.')
plt.ylabel(r'$P$ = pressure  / $P_\odot$')

plt.subplot(414)
plt.plot(r/Rsun, T/Tsun)
plt.plot(r[n]/Rsun, T[n]/Tsun, 'r.')
plt.ylabel(r'$T$ = temperature  / $T_\odot$')

plt.xlabel(r'$r$ = distance from center  / $R_\odot$')
plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()

