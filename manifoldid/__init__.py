import numpy as np
import scipy.interpolate as interpolate
from scipy.integrate import ode
import .cdcl

def Glider(t, v, theta=0, C=Cplate):
  #v = [vx, vz]
  psi = -np.arctan2(v[1], v[0])
  Yp = [(v[0]**2+v[1]**2)*(C(psi + theta)[1]*np.sin(psi)-C(psi + theta)[0]*np.cos(psi)), (v[0]**2+v[1]**2)*(C(psi + theta)[1]*np.cos(psi)+C(psi + theta)[0]*np.sin(psi))-1]
  return np.array(Yp)
  
def GliderAuto(v, theta=0, C=Cplate):
  #v = [vx, vz]
  psi = -np.arctan2(v[1], v[0])
  Yp = [(v[0]**2+v[1]**2)*(C(psi + theta)[1]*np.sin(psi)-C(psi + theta)[0]*np.cos(psi)), (v[0]**2+v[1]**2)*(C(psi + theta)[1]*np.cos(psi)+C(psi + theta)[0]*np.sin(psi))-1]
  return np.array(Yp)
  
def bisectInt(func, guess, *args):
  r = ode(func).set_integrator('lsoda', atol=10**(-8), rtol=10**(-8))
  # func is the function that we want to use the bisection method on
  r.set_f_params(*args)
  t0 = 0; dt = -0.001; tf = -20
  eps=2.5 
  Vx0 = guess[0]
  Min = guess[1]-eps
  Max = guess[1]+eps
  while Max-Min > 10**(-8):
    Y = []; T = []
    Mid = Min+(Max-Min)/2
    y0 = np.array([Vx0, Mid])
    r.set_initial_value(y0, t0)
    test = -2
    test2 = 0
    while r.successful() and test < 0 and test > -5 and abs(test2) < 100 and r.t > tf:
      r.integrate(r.t+dt)
      Y.append(r.y)
      T.append(r.t)
      test = r.y[1]
      test2 = r.y[0]
    assert r.successful()
    if Y[-1][1] > Mid:
      Max = Mid
    else:
      Min = Mid
  return np.array([guess[0], Min + (Max-Min)/2])
  
def intDiff(func, y0, *args):
  dt = 0.0001
  r = ode(func).set_integrator('lsoda', atol=10**(-8), rtol=10**(-8))
  r.set_f_params(*args)
  t0 = 0; dt = dt
  Y = []; T = []
  r.set_initial_value(y0, t0)
  while r.successful() and r.t < 2*dt:
    r.integrate(r.t+dt)
    Y.append(r.y)
    T.append(r.t)
  return np.sign(Y[1][0]-Y[0][0])*np.sqrt((Y[1][1]-Y[0][1])**2+(Y[1][0]-Y[0][0])**2)/dt
  
def eigType(eig):
  if np.imag(eig[0]) < 0.0001:
    if np.abs(eig[0]) < 0.01 or np.abs(eig[1]) < 0.01:
      color=(0, 0, 0)
    elif eig[0] < 0 and eig[1] < 0:
      color=(0, 0, 1)
    elif eig[0]*eig[1] < 0:
      color=(0, 1, 0)
    else:
      color = (1, 0, 0)
  else:
    if np.real(eig[0]) < 0:
      color= (0, 1, 1)
    else:
      color= (1, 0, 1)
  return color
