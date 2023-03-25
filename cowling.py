# refereces:
#   M. Schwarzschild
#    "Structure and Evolution of the Stars" (SES)
#   R. Kippenhahn and A. Weigert
#    "Stellar Structure and Evolution" (SSE)

import numpy as np
from scipy.integrate import odeint,solve_bvp,cumtrapz
from constants import kB,mH,stefan,Grav,eV

a,b = 1, 3.5 # slope of Kramer's opacity
nu = 16 # slope of nuclear reaction rate
kappa0 = 4.34e24 # coeff of Kramer's opacity (cgs)
eps0 = 2.67e-114 # coeff of nuclear reaction rate (cgs)
q_CN = 25.02e6*eV/4/mH # energy generated in CNO cycle

EPS = 1e-4 # parameter to avoid singularity at surface
a1,a2,b3,b4 = a+1,a+2,b+3,b+4
a1b4 = a1/(a+b4)
kmG = np.log(kB/mH/Grav) # log to avoid overflow
C0 = np.log(3/16*kappa0/stefan/(4*np.pi)**a2) + b4*kmG
B0 = np.log(eps0/4/np.pi) - nu*kmG # SSE eqs.(21.13)(21.21)

def kappa(rho,T,X,Z): # Kramer's opacity
    return kappa0*Z*(1+X)*rho**a/T**b

def epsilon(rho,T,X,Z): # nuclear reaction rate
    return eps0*Z*X*rho*T**nu

def RadEnv(x,C):# radiative envelope
    def diff_eq(y,x):# equation of stellear structure
        q,p,t = y
        if t<EPS:# avoid singularity at surface
            pt = (a1b4*t**b3/C)**(1/a1) # SSE eq.(21.15)
            x2 = x**2
            dq = pt*x2
            dp = -q*pt/x2
            dt = -a1b4/x2
        else:
            x2,pt = x**2,p/t
            dq = x2*pt
            dp = -q*pt/x2
            dt = -C*pt**a1/x2/t**b3
        return dq,dp,dt

    if np.isscalar(x):
        y = odeint(diff_eq, [1,0,0], [1,x])
        return y[-1]
    else:
        return odeint(diff_eq, [1,0,0], x)

class CowlingModel:
    """ Cowling model for steller evolution """
    def __init__(self, Z, X, X1=None,
                 init=None, N=128, N1=128):
        """ X,Z = mass fractions of hydrogen and metals
        (Y = 1-X-Z = mass fraction of helium)
        X  = mass fractions of hydrogen in envelope
        X1 = mass fractions of hydrogen in core
        init = initial guess of solution as CowlingModel object
        N = number of mesh points in envelope
        N1 = number of mesh points in core
        if init is not None, N and N1 are not used
        """
        if X1 is None: X1 = X # homogeneous star at age zero
        mu = 4/(3 + 5*X - Z) # mean molecular weight
        mu1 = 4/(3 + 5*X1 - Z)
        mm1 = mu1/mu
        n1 = 0.4/mm1**a*(1+X)/(1+X1) # SES eq.(20.2)

        def diff_eq(x,y,p):# Lane-Emden equation (n=1.5)
            zf2 = p[0]**2
            d2w = -2*y[1,1:]/x[1:] - y[0,1:]**1.5*zf2
            return y[1], np.r_[-zf2/3, d2w]

        def bc(ya,yb,p):# boundary condition
            zf,xf,C = p
            q,p,t = RadEnv(xf,C)
            pt = p/t
            nabla = C*pt**a1/t**b3/q
            U = xf**3*pt/q
            V = q/t/xf

            w0,dw0 = ya
            w,dw = yb
            U1 = -w**1.5*zf**2/dw
            V1 = -2.5*dw/w

            return (w0-1, dw0,
                    U1/U - mm1,
                    V1/V - mm1,
                    nabla - n1)

        if init is None:
            x = np.linspace(0,1,N1)
            z = x*1.2
            y = [1-z**2/6, -z/3]
            p = np.r_[1.2, 0.2, 1e-6]
        elif isinstance(init, CowlingModel):
            s = init.sol
            x,y,p = s.x,s.y,s.p
        else:
            raise RuntimeError('bad init')

        s = solve_bvp(diff_eq, bc, x,y,p)
        zf,xf,C = s.p
        z,w,dw = s.x*zf, s.y[0], s.y[1]/zf
        wf,dwf = w[-1],dw[-1]
        B = 1/np.trapz(z**2*w**(nu+3), z) # SSE eq.(21.22)

        x = np.linspace(1,xf,N)
        q,p,t = RadEnv(x,C).T
        tf,pf,qf = t[-1],p[-1],q[-1]

        # SSE eqs.(21.13)(21.21)
        C1 = Z*(1+X)/mu**b4
        B1 = Z*X1*(mu*tf/wf)**nu*(xf/zf/wf)**3*(pf/tf*mm1)**2

        z,w,dw = s.x, w/wf, dw/dwf
        # outputs as properties of CowlingModel object
        self.x = np.r_[z*xf, x[::-1]] # normalized radial coord.
        self.t = np.r_[w*tf, t[::-1]] # ...  temperature
        self.p = np.r_[w**2.5*pf, p[::-1]] # ... pressure
        self.q = np.r_[z**2*dw*qf, q[::-1]] # ... mass
        self.C = np.log(C/C1) - C0 # fitting condition 1
        self.B = np.log(B/B1) - B0 # fitting condition 2
        self.X = np.r_[[X1]*len(z), [X]*N] # hydrogen fraction
        self.Z = Z # metal fraction
        self.Nf = len(z) # index of fitting point
        self.sol = s # solution of Lane-Emden equation

    def RLT(self, M):
        """ M = mass of star / gram
        return R,L,T (each same shape as M) where
          R = radius / cm, L = luminosity / erg/s
          T = effective (surface) temperature / K
        M can be array
        """
        B,C = self.B, self.C
        lnM = np.log(M) # log to avoid overflow
        lnR = (B+C - (a2-b3+nu)*lnM)/(b-3*a1-nu)
        lnL = (nu+2)*lnM - (nu+3)*lnR - B
        R,L = np.exp(lnR), np.exp(lnL)
        T = (L/4/np.pi/R**2/stefan)**0.25
        return R,L,T

    def set_mass(self, M):
        """ M must be scalar """
        R,L,T = self.RLT(M)
        self.M = M # mass / g
        self.R = R # radius / cm
        self.L = L # luminosity / erg/s
        self.T = T # surface temperature / K

    def rmlPTrho(self):
        """ return radial profiles (each 1d array)
        r = distance from center / cm
        m = mass with sphere of radius r / gram
        l = energy flow at r / erg/s
        P = pressure at r / dyne/cm^2
        T = temperature at r / K
        rho = density at r / gram/cm^3
        assume set_mass(M) has been done already
        """
        if not hasattr(self, 'M'):
            raise RuntimeError('mass is not set')
        M,R,L = self.M, self.R, self.L
        X,Z = self.X, self.Z
        mu = 4/(3 + 5*X - Z)
        P = Grav*M**2/4/np.pi/R**4*self.p
        T = Grav*M*mu[-1]*mH/kB/R*self.t
        r,m = R*self.x, M*self.q
        rho = np.r_[mu[:-1]*mH*P[:-1]/kB/T[:-1], 0]
        e = epsilon(rho,T,X,Z)
        l = cumtrapz(4*np.pi*rho*r**2*e, r, initial=0)
        return r,m,l,P,T,rho

    def del_rad(self):# dlnT/dlnP (1d array)
        r,m,l,P,T,rho = self.rmlPTrho()
        X,Z = self.X[1:-1], self.Z
        rho,T = rho[1:-1],T[1:-1]
        lmP = l[1:-1]/m[1:-1]*P[1:-1]
        k = kappa(rho,T,X,Z)
        dT = 3/64/np.pi/stefan/Grav*k*lmP/T**4
        del_rad = np.r_[dT[0], dT, dT[-1]]
        return del_rad

    def dX_dt(self):# hydrogen comsumption rate / sec^-1
        if not hasattr(self, 'M'):
            raise RuntimeError('mass is not set')
        M,L,q,n = self.M, self.L, self.q, self.Nf
        return -L/(q_CN * M * q[n])
