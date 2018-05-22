"""Some example dynamical systems

"""

from numpy import array, cos, sin, pi, tanh, arctan2

def kevrekidis(y, eps=0.01):
  return array([-y[0]-y[1]+2, 1/eps*(y[0]**3-y[1])])

def ex11(y):
  return -array([tanh(y[0]**2/4)+y[0],y[0]+2*y[1]])

def rotHoop(y, eps=0.1, gamma=2.3):
  return np.array([y[1], 1/eps*(np.sin(y[0])*(gamma*np.cos(y[0])-1)-y[1])])

# For eps=0.01 a=0.57662277221679688 works nicely
def vanderPol(y, eps=0.01, a=0.575):
  return np.array([1/eps*(y[1]-y[0]**3+y[0]), a-y[0]])

def Csquirrel(alpha):
  return 1, 1.5+2*cos(2*alpha)
def Cplate(alpha):
  return 1.4-cos(2*alpha), 1.2*sin(2*alpha)
def Cball(alpha):
  return array([2, 0])

def GliderAuto(v, theta=-pi/18, C=Cplate):
  #v = [vx, vz]
  psi = -arctan2(v[1], v[0])
  Yp = [(v[0]**2+v[1]**2)*(C(psi + theta)[1]*sin(psi)-C(psi + theta)[0]*cos(psi)), (v[0]**2+v[1]**2)*(C(psi + theta)[1]*cos(psi)+C(psi + theta)[0]*sin(psi))-1]
  return array(Yp)
