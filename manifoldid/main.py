#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:54:40 2017

@author: gknave
"""

# %%
import numpy as np
from numpy.ma import MaskedArray
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.integrate import ode
from matplotlib import cm
#from .testfunctions import all

def goodfigure(xlims, ylims, area=130):
  """Creates a new (good) figure
  
  This is Gary's attempt to make a better looking figure.
  It will open a 2-dimensional figure that matches the aspect ratio 
  of the axes, with aspect ratio set to 'equal.'
  
  Parameters
  ----------
  xlims : length 2 list or tuple of floats
    The x-axis limits of the figure
  
  ylims : length 2 list or tuple of floats
    The y-axis limits of the figure
  
  area : integer or float, optional, default: 130
    The area in inches of the figure
    If figsize=(a, b), area=a*b
  
  Returns
  -------
  A matplotlib.pyplot Figure instance with axes set to xlims and ylims, with a matching aspect ratio
  
  Examples
  --------
  goodfigure([-2, 2], [-3, 3], area=100) # Returns a figure of figsize=(8, 12)
  
  """
  from matplotlib.pyplot import figure, gca, xlim, ylim
  from matplotlib import rc
  from seaborn import set_style
  from numpy import sqrt
  rc('xtick', labelsize=22)
  rc('ytick', labelsize=22)
  set_style('ticks')

  ar = (ylims[1]-ylims[0])/(xlims[1]-xlims[0])
  w = sqrt(area//ar)//1
  figure(figsize=(w, ar*w))
  gca().set_aspect('equal', adjustable='box', anchor='C')
  xlim(xlims)
  ylim(ylims)

def autonomous_odeint(func, y0, *fargs, t0=0, dt=0.01, tf=200, ret_success=False, stiff=True):
  dt = np.abs(dt)*np.sign(tf)
  def odefun(t, v):
    return func(v, *fargs)
  if stiff:
    r = ode(odefun).set_integrator('vode', atol=10**(-8), rtol=10**-8, nsteps=50000, method='bdf')
  else:
    r = ode(odefun).set_integrator('lsoda', atol=10**(-8), rtol=10**(-8))
  r.set_initial_value(y0, t0)
  Y = []; T = []; Y.append(y0); T.append(t0)
  while r.successful() and r.t/tf < 1:
    r.integrate(r.t+dt)
    Y.append(r.y)
    T.append(r.t)
  if ret_success:
    return T, Y, r.successful()
  return np.array(T), np.array(Y)


def s1(func, xlims, ylims, ds, *fargs, output=False, masked=False, plot=True, cmap='coolwarm', newfig=True, savefig=False, figname='localRho.pdf', vmin=-3, vmax=3):
  """The trajectory-normal repulsion rate

  This function finds the trajectory-normal repulsion rate field introduced by Nave and Ross, 2017.
  Gives a measure of how much the trajectory passing through x_0 attracts or repels nearby
  trajectories infinitesimally.

  Parameters
  ----------
  func : function
    A two-dimensional function f([x, y]) which returns [f1, f2]

  xlims : length 2 list or tuple of floats
    The x-axis limits of the figure

  ylims : length 2 list or tuple of floats
    The y-axis limits of the figure

  ds : float
    Grid spacing in both directions, assumed to be equal

  *fargs : arguments to pass to func

  output : boolean, optional, default: False
    If set to True, returns outputs described below

  plot : boolean, optional, default: True

  cmap : colormap, optional, default: 'bone'
    Selection of colormap from matplotlib.cmap

  newfig : boolean, optional, default: True
    Chooses whether phase_plot is plotted in a new figure. To put phase_plot
    on top of an existing figure, set to False

  savefig : boolean, optional, default: False
    Chooses whether to save the figure as an image file, named with figname
    Uses matplotlib.pyplot.savefig

  figname : string, optional, default: 'repulsion_factor.png'
    If savefig=True is used, specifies the name of the imagefile within the
    matplotlib.pyplot.savefig command

  Returns
  -------
  If plot=True, returns a matplotlib.pyplot.pcolormesh instance

  If output=True, the following variables are returned:

  x1 : 1-dimensional numpy.np.array
    This represents the points along the x-axis

  x2 : 1-dimensional numpy.np.array
    This represents the points along the y-axis
    To generate all points, use numpy.np.meshgrid(x1, x2)

  rho_dot : scalar field
    The trajectory-normal repulsion factor given by
    rho_dot = <n_T, \nabla F^T n_0>


  """
  if plot and newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      U[n, m], V[n, m] = func(y0, *fargs)
  [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
  s1 = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      S = 0.5*(Grad + np.transpose(Grad))
      vals= LA.eigvals(S)
      s1[n, m] = vals[np.argmax(np.abs(vals))]
  if masked:
    rho_dot = MaskedArray(s1 >= 0, s1)
  if plot:
    lim = np.max(np.abs(s1))
    ax = plt.gca()
    mesh = ax.pcolormesh(X1, X2, s1, cmap=cmap, vmin=-lim, vmax=lim)
    clb = plt.colorbar(mesh)
    clb.ax.set_title('$s_2$', fontsize=28, y=1.02)
    plt.xlim(xlims); plt.ylim(ylims)
    if savefig:
      plt.savefig(figname, transparent=True, bbox_inches='tight')
  if output:
    return x1, x2, s1


def s2(func, xlims, ylims, ds, *fargs, output=False, masked=False, plot=True, cmap='coolwarm', newfig=True, savefig=False, figname='localRho.pdf', vmin=-3, vmax=3):
  """The trajectory-normal repulsion rate

  This function finds the trajectory-normal repulsion rate field introduced by Nave and Ross, 2017.
  Gives a measure of how much the trajectory passing through x_0 attracts or repels nearby
  trajectories infinitesimally.

  Parameters
  ----------
  func : function
    A two-dimensional function f([x, y]) which returns [f1, f2]

  xlims : length 2 list or tuple of floats
    The x-axis limits of the figure

  ylims : length 2 list or tuple of floats
    The y-axis limits of the figure

  ds : float
    Grid spacing in both directions, assumed to be equal

  *fargs : arguments to pass to func

  output : boolean, optional, default: False
    If set to True, returns outputs described below

  plot : boolean, optional, default: True

  cmap : colormap, optional, default: 'bone'
    Selection of colormap from matplotlib.cmap

  newfig : boolean, optional, default: True
    Chooses whether phase_plot is plotted in a new figure. To put phase_plot
    on top of an existing figure, set to False

  savefig : boolean, optional, default: False
    Chooses whether to save the figure as an image file, named with figname
    Uses matplotlib.pyplot.savefig

  figname : string, optional, default: 'repulsion_factor.png'
    If savefig=True is used, specifies the name of the imagefile within the
    matplotlib.pyplot.savefig command

  Returns
  -------
  If plot=True, returns a matplotlib.pyplot.pcolormesh instance

  If output=True, the following variables are returned:

  x1 : 1-dimensional numpy.np.array
    This represents the points along the x-axis

  x2 : 1-dimensional numpy.np.array
    This represents the points along the y-axis
    To generate all points, use numpy.np.meshgrid(x1, x2)

  rho_dot : scalar field
    The trajectory-normal repulsion factor given by
    rho_dot = <n_T, \nabla F^T n_0>


  """
  if plot and newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      U[n, m], V[n, m] = func(y0, *fargs)
  [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
  s2 = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      S = 0.5*(Grad + np.transpose(Grad))
      vals= LA.eigvals(S)
      s2[n, m] = vals[np.argmin(np.abs(vals))]
  if masked:
    rho_dot = MaskedArray(rho_dot >= 0, rho_dot)
  if plot:
    lim = np.max(np.abs(s2))
    ax = plt.gca()
    mesh = ax.pcolormesh(X1, X2, s2, cmap=cmap, vmin=-lim, vmax=lim)
    clb = plt.colorbar(mesh)
    clb.ax.set_title('$s_2$', fontsize=28, y=1.02)
    plt.xlim(xlims); plt.ylim(ylims)
    if savefig:
      plt.savefig(figname, transparent=True, bbox_inches='tight')
  if output:
    return x1, x2, s2
    
def phase_plot(func, xlims, ylims, *fargs, color=(0.5, 0.75, 0.6), paths=True, newfig=True, savefig=False, figname='phase_plot.png'):
  """Generate a phase portrait of an autonomous function
  
  Given an autonomous two-dimensional function and the limits, returns a quiver plot of the vector field

  Parameters
  ----------
  func : function
    A two-dimensional function f([x, y]) which returns [f1, f2]
  
  xlims : length 2 list or tuple of floats
    The x-axis limits of the figure
  
  ylims : length 2 list or tuple of floats
    The y-axis limits of the figure

  *fargs : arguments to pass to func

  color : tuple or string, optional, default: (0.5, 0.75, 0.6)
    Sets the color of the vector arrows in the phase portrait
  
  paths : boolean, optional, default: True
    Chooses whether to plot example trajectories using advect_trajectories

  newfig : boolean, optional, default: True
    Chooses whether phase_plot is plotted in a new figure. To put phase_plot
    on top of an existing figure, set to False

  savefig : boolean, optional, default: False
    Chooses whether to save the figure as an image file, named with figname
    Uses matplotlib.pyplot.savefig

  figname : string, optional, default: 'phase_plot.png'
    If savefig=True is used, specifies the name of the imagefile within the
    matplotlib.pyplot.savefig command

  Returns
  -------
  A quiver plot of the velocity vectors in a two-dimensional ODE.

  """
  if newfig:
    goodfigure(xlims, ylims)
  area = 150
  ar = (ylims[1]-ylims[0])/(xlims[1]-xlims[0])
  w = np.sqrt(area//ar)//1
  x1 = np.linspace(xlims[0], xlims[1], w+1)
  x2 = np.linspace(ylims[0], ylims[1], w*ar+1)
  X1, X2 = np.meshgrid(x1, x2)
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      Y = [x1[m], x2[n]]
      Ytemp = func(Y, *fargs)
      U[n, m] = Ytemp[0]
      V[n, m] = Ytemp[1]
  plt.quiver(X1, X2, U, V, color=color)
  plt.xlim(xlims); plt.ylim(ylims)
  if paths:
    advect_trajectories(func, xlims, ylims, *fargs, newfig=False)
  if savefig:
    plt.savefig(figname, transparent=True, bbox_inches='tight')
 

def advect_trajectories(func, xlims, ylims, *fargs, color=(0.8, 0.7, 0.5), newfig=True, linewidth=2, offset=1, N=11, **kwargs):
  """Visualize trajectories in phase space
  Advects points for an autonomous two-dimensional function, func, from
  outside of the window specified by xlims and ylims forward in time

  Parameters
  ----------
  func : function
    A two-dimensional function f([x, y]) which returns [f1, f2]
  
  xlims : length 2 list or tuple of floats
    The x-axis limits of the figure
  
  ylims : length 2 list or tuple of floats
    The y-axis limits of the figure

  *fargs : arguments to pass to func

  color : tuple or string, optional, default: (0.5, 0.75, 0.6)
    Sets the color of the vector arrows in the phase portrait

  newfig : boolean, optional, default: True
    Chooses whether advect_trajectories is plotted in a new figure. To put 
    advect_trajectories on top of an existing figure, set to False

  linewidth : float or integer, optional, default: 2
    Specifies the width of the lines plotted using matplotlib.pyplot.plot
  
  offset : float or integer, optional, default: 1
    Distance outside of the limits specified by xlims and ylims that trajectories begin

  N : integer, optional, default: 11
    Number of trajectories along each side of the figure, 4*N trajectories are shown in total

  **kwargs : keyword arguments
    Integration arguments to be passed to autonomous_odeint

  Returns
  -------
  A collection of matplotlib.pyplot.line instances created with matplotlib.pyplot.plot from
  around the figure limits given by xlims, ylims

  """
  if newfig:
    goodfigure(xlims, ylims)
  xlims = [xlims[0]-offset, xlims[1]+offset]
  ylims = [ylims[0]-offset, ylims[1]+offset]
  for l in np.linspace(xlims[0], xlims[1], N):
    for k in ylims:
      T, Y = autonomous_odeint(func, [l, k], *fargs, **kwargs)
      Y = np.array(Y)
      plt.plot(Y[:, 0], Y[:, 1], c=color, linewidth=linewidth)
  for l in np.linspace(ylims[0], ylims[1], N):
    for k in xlims:
      T, Y = autonomous_odeint(func, [k, l], *fargs, **kwargs)
      Y = np.array(Y)
      plt.plot(Y[:, 0], Y[:, 1], c=color, linewidth=linewidth)


def peelingOff(func, xlims, ylims, *fargs, flip=False, newfig=False, testlims=[-50, 50], color=(0.7, 0.4, 0.1), linewidth=3):
  """Attracting manifold detection using the method of peeling off

  The method of peeling off is a bisection method which integrates a fuction backward in time
  to discover an attracting manifold. It is assumed that there is a curve which separates points
  beginning at testlims[0] from those beginning at testlims[1]. This function finds the points of the attracting
  manifold at xlims (or ylims, if flip=True) and integrates them forward to find the attracting manifold.

  See {publication} for details.


  Parameters
  ----------
  func : function
    A two-dimensional function f([x, y]) which returns [f1, f2]
  
  xlims : length 2 list or tuple of floats
    The x-axis limits of the figure
  
  ylims : length 2 list or tuple of floats
    The y-axis limits of the figure

  *fargs : arguments to pass to func

  flip : boolean, optional, default: False
    This parameter determines whether the manifold lies between ylims (flip=False) or
    between xlims (flip=True).

  newfig : boolean, optional, default: True
    Chooses whether peelingOff is plotted in a new figure. To put 
    peelingOff on top of an existing figure, set to False

  testlims : length 2 list or tuple of floats
    These are the limits of the test window, once the backward integration has reached
    these limits, the function decides whether y0 lies above or below the attracting
    manifold.

  Returns
  -------
  A pair of curves beginning at xlims and ending at a fixed point in the middle which
  represents an attracting manifold in the system.
  
  """
  if newfig:
    goodfigure(xlims, ylims)
  def odefun(t, v):
    return func(v, *fargs)
  r = ode(odefun).set_integrator('lsoda', atol=10**(-8), rtol=10**(-8))
  if not flip:
    rng = xlims
    ind = 1
  else:
    rng = ylims
    ind = 0
  for x0 in rng:
    dt = -0.001; tf = -200
    if not flip:
      Min, Max = ylims
    else:
      Min, Max = xlims
    while Max-Min > 10**(-8):
      Y = []; T = []
      Mid = Min+(Max-Min)/2
      if not flip:
        Y0 = [x0, Mid]
      else:
        Y0 = [Mid, x0]
      r.set_initial_value(Y0, 0)
      test = np.mean(testlims)
      while r.successful() and test > testlims[0] and test < testlims[1] and abs(r.t/tf) < 1:
        r.integrate(r.t+dt)
        Y.append(r.y)
        T.append(r.t)
        test = r.y[ind]
      if r.successful():
        if Y[-1][ind] > Mid:
          Max = Mid
        else:
          Min = Mid
      else:
        if Y[-2][ind] > Mid:
          Max = Mid
        else:
          Min = Mid
    dt = 0.01; tf = 40
    if flip:
      Y0 = [Mid, x0]
    else:
      Y0 = [x0, Mid]
    T, Y = autonomous_odeint(func, Y0, *fargs)
    plt.plot(Y[:, 0], Y[:, 1], color=color, linewidth=linewidth)

def s1(func, xlims, ylims, ds, *fargs, output=False, masked=False, plot=True, cmap='coolwarm', newfig=True, savefig=False, figname='localRho.pdf', vmin=None, vmax=None):
  if plot and newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      U[n, m], V[n, m] = func(y0, *fargs)
  [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
  s1 = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      S = 0.5*(Grad + np.transpose(Grad))
      vals= LA.eigvals(S)
      s1[n, m] = vals[np.argmax(vals)]
  if masked:
    rho_dot = MaskedArray(s1 >= 0, s1)
  if plot:
    if vmin==vmax:
      lim = np.max(np.abs(s1))
      vmin = -lim
      vmax = lim
    ax = plt.gca()
    mesh = ax.pcolormesh(X1, X2, s1, cmap=cmap, vmin=vmin, vmax=vmax)
    clb = plt.colorbar(mesh)
    clb.ax.set_title('$s_1$', fontsize=28, y=1.02)
    plt.xlim(xlims); plt.ylim(ylims)
    if savefig:
      plt.savefig(figname, transparent=True, bbox_inches='tight')
  if output:
    return x1, x2, s1

def s2(func, xlims, ylims, ds, *fargs, output=False, masked=False, plot=True, cmap='coolwarm', newfig=True, savefig=False, figname='localRho.pdf', vmin=None, vmax=None):
  if plot and newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      U[n, m], V[n, m] = func(y0, *fargs)
  [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
  s2 = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      S = 0.5*(Grad + np.transpose(Grad))
      vals= LA.eigvals(S)
      s2[n, m] = vals[np.argmin(vals)]
  if masked:
    rho_dot = MaskedArray(s2 >= 0, s2)
  if plot:
    if vmin==vmax:
      lim = np.max(np.abs(s2))
      vmin = -lim
      vmax = lim
    ax = plt.gca()
    mesh = ax.pcolormesh(X1, X2, s2, cmap=cmap, vmin=vmin, vmax=vmax)
    clb = plt.colorbar(mesh)
    clb.ax.set_title('$s_1$', fontsize=28, y=1.02)
    plt.xlim(xlims); plt.ylim(ylims)
    if savefig:
      plt.savefig(figname, transparent=True, bbox_inches='tight')
  if output:
    return x1, x2, s2

def curvatureField(func, xlims, ylims, ds, *fargs, plot=True, cmap='PRGn', newfig=True, vmin=None, vmax=None):
  if plot and newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      U[n, m], V[n, m] = func(y0, *fargs)
  [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
  kappa = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      Uhat = np.array([V[n, m], -U[n, m]])
      Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      kappa[n, m] = np.dot(Uhat, np.dot(Grad, Utemp))/np.dot(Utemp, Utemp)**(1.5)
  if plot:
    if vmin==vmax:
      lim = np.max(np.abs(kappa))
      vmin = -lim
      vmax = lim
    ax = plt.gca()
    mesh = ax.pcolormesh(X1, X2, kappa, cmap=cmap, vmin=vmin, vmax=vmax)
    clb = plt.colorbar(mesh)
    clb.ax.set_title('$\\kappa$', fontsize=28, y=1.02)
    plt.xlim(xlims); plt.ylim(ylims)
  return x1, x2, kappa

def repulsion_rate(func, xlims, ylims, ds, T, *fargs, output=False, plot=True, cmap='bone', newfig=True, savefig=False, figname='repulsion_factor.png', **kwargs):
  """The trajectory-normal repulsion factor

  This function finds the trajectory-normal repulsion factor field introduced by Haller, 2011 
  as the trajectory-normal repulsion rate, rho_T(x_0), and renamed by Nave and Ross, 2017.
  Gives a measure of how much the trajectory passing through x_0 attracts or repels nearby
  trajectories after time T.

  Parameters
  ----------
  func : function
    A two-dimensional function f([x, y]) which returns [f1, f2]
  
  xlims : length 2 list or tuple of floats
    The x-axis limits of the figure
  
  ylims : length 2 list or tuple of floats
    The y-axis limits of the figure

  ds : float
    Grid spacing in both directions, assumed to be equal

  T : float
    Integration time to generate rho_T

  *fargs : arguments to pass to func
  
  output : boolean, optional, default: False
    If set to True, returns outputs described below

  plot : boolean, optional, default: True

  cmap : colormap, optional, default: 'bone'
    Selection of colormap from matplotlib.cmap

  newfig : boolean, optional, default: True
    Chooses whether phase_plot is plotted in a new figure. To put phase_plot
    on top of an existing figure, set to False

  savefig : boolean, optional, default: False
    Chooses whether to save the figure as an image file, named with figname
    Uses matplotlib.pyplot.savefig

  figname : string, optional, default: 'repulsion_factor.png'
    If savefig=True is used, specifies the name of the imagefile within the
    matplotlib.pyplot.savefig command

  Returns
  -------
  If plot=True, returns a matplotlib.pyplot.pcolormesh instance

  If output=True, the following variables are returned:

  x1 : 1-dimensional numpy.np.array
    This represents the points along the x-axis

  x2 : 1-dimensional numpy.np.array
    This represents the points along the y-axis
    To generate all points, use numpy.np.meshgrid(x1, x2)
  
  rho_T : scalar field
    The trajectory-normal repulsion factor given by
    rho_T = <n_T, \nabla F^T n_0>


  """
  if newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  yOut = np.zeros(np.concatenate(((2,), np.shape(U))))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      U[n, m], V[n, m] = func(y0, *fargs)
      time, Y = autonomous_odeint(func, y0, *fargs, tf=T, **kwargs)
      yOut[:, n, m] = Y[-1, :]
  [DUy, DUx] = np.gradient(yOut[0, :, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(yOut[1, :, :], ds, edge_order=2)
  rho = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      DF = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      CG = np.dot(np.transpose(DF), DF)
      if abs(LA.det(CG)) < 10**(-15):
        rho[n, m] = 0
      else:
        rho[n, m] = np.sqrt(abs(np.dot(Utemp, Utemp)*LA.det(CG)/np.dot(Utemp, np.dot(CG, Utemp))))
  if plot:
    plt.pcolormesh(X1, X2, rho, cmap=cmap, vmin=0, vmax=6)
    plt.xlim(xlims); plt.ylim(ylims)
    if savefig:
      plt.savefig(figname, transparent=True, bbox_inches='tight')
  if output:
    return x1, x2, rho

def ftle_field(func, xlims, ylims, ds, T, *fargs, dt=0.01, output=False, plot=True, cmap='bone', newfig=True, savefig=False, figname='ftle_field.png', **kwargs):
  """Calculate the finite-time Lyapunov exponent field

  The finite-time Lyapunov exponent is a measure of the stretching between nearby
  trajectories in a flow over a finite time [0, T]
  
  Parameters
  ----------
  func : function
    A two-dimensional function f([x, y]) which returns [f1, f2]
  
  xlims : length 2 list or tuple of floats
    The x-axis limits of the figure
  
  ylims : length 2 list or tuple of floats
    The y-axis limits of the figure

  ds : float
    Grid spacing in both directions, assumed to be equal

  T : float
    Integration time to generate rho_T

  *fargs : arguments to pass to func
  
  dt : float
    The time step passed to autonomous_odeint

  output : boolean, optional, default: False
    If set to True, returns outputs described below

  plot : boolean, optional, default: True

  cmap : colormap, optional, default: 'bone'
    Selection of colormap from matplotlib.cmap

  newfig : boolean, optional, default: True
    Chooses whether phase_plot is plotted in a new figure. To put phase_plot
    on top of an existing figure, set to False

  savefig : boolean, optional, default: False
    Chooses whether to save the figure as an image file, named with figname
    Uses matplotlib.pyplot.savefig

  figname : string, optional, default: 'repulsion_factor.png'
    If savefig=True is used, specifies the name of the imagefile within the
    matplotlib.pyplot.savefig command

  Returns
  -------
  If plot=True, returns a matplotlib.pyplot.pcolormesh instance

  If output=True, the following variables are returned:

  x1 : 1-dimensional numpy.np.array
    This represents the points along the x-axis

  x2 : 1-dimensional numpy.np.array
    This represents the points along the y-axis
    To generate all points, use numpy.np.meshgrid(x1, x2)
  
  ftle : scalar field
    The finite-time Lyapunov exponent field
  """
  if plot and newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  yOut = np.zeros(np.concatenate(((2,), np.shape(X1))))
  ftle = np.zeros(np.shape(X1))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      time, Y = autonomous_odeint(func, y0, *fargs, dt = dt, tf=T, **kwargs)
      yOut[:, n, m] = Y[-1, :]
  [DUy, DUx] = np.gradient(yOut[0, :, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(yOut[1, :, :], ds, edge_order=2)
  for m in range(len(x1)):
    for n in range(len(x2)):
      DF = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      CG = np.dot(np.transpose(DF), DF)
      lamdas = LA.eigvals(CG)
      lamdamax = lamdas[np.argmax(lamdas)]
      ftle[n, m] = 1/abs(T)*np.sqrt(lamdamax)
  if plot:
    plt.pcolormesh(X1, X2, ftle, cmap=cmap)
    plt.xlim(xlims); plt.ylim(ylims)
    if savefig:
      plt.savefig(figname, transparent=True, bbox_inches='tight')
  if output:
    return x1, x2, ftle

def divergence_rate(func, xlims, ylims, ds, *fargs, output=False, masked=False, plot=True, cmap='coolwarm', newfig=True, savefig=False, figname='localRho.pdf', vmin=None, vmax=None):
  """The trajectory-normal repulsion rate

  This function finds the trajectory-normal repulsion rate field introduced by Nave and Ross, 2017.
  Gives a measure of how much the trajectory passing through x_0 attracts or repels nearby
  trajectories infinitesimally.

  Parameters
  ----------
  func : function
    A two-dimensional function f([x, y]) which returns [f1, f2]
  
  xlims : length 2 list or tuple of floats
    The x-axis limits of the figure
  
  ylims : length 2 list or tuple of floats
    The y-axis limits of the figure

  ds : float
    Grid spacing in both directions, assumed to be equal

  *fargs : arguments to pass to func
  
  output : boolean, optional, default: False
    If set to True, returns outputs described below

  plot : boolean, optional, default: True

  cmap : colormap, optional, default: 'bone'
    Selection of colormap from matplotlib.cmap

  newfig : boolean, optional, default: True
    Chooses whether phase_plot is plotted in a new figure. To put phase_plot
    on top of an existing figure, set to False

  savefig : boolean, optional, default: False
    Chooses whether to save the figure as an image file, named with figname
    Uses matplotlib.pyplot.savefig

  figname : string, optional, default: 'repulsion_factor.png'
    If savefig=True is used, specifies the name of the imagefile within the
    matplotlib.pyplot.savefig command

  Returns
  -------
  If plot=True, returns a matplotlib.pyplot.pcolormesh instance

  If output=True, the following variables are returned:

  x1 : 1-dimensional numpy.np.array
    This represents the points along the x-axis

  x2 : 1-dimensional numpy.np.array
    This represents the points along the y-axis
    To generate all points, use numpy.np.meshgrid(x1, x2)
  
  rho_dot : scalar field
    The trajectory-normal repulsion factor given by
    rho_dot = <n_T, \nabla F^T n_0>


  """
  if plot and newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      U[n, m], V[n, m] = func(y0, *fargs)
  [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
  rho_dot = np.zeros(np.shape(U))
  J = np.array([[0, 1], [-1, 0]])
  for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      S = 0.5*(Grad + np.transpose(Grad))
      rho_dot[n, m] = np.dot(Utemp, np.dot(np.dot(np.transpose(J), np.dot(S, J)), Utemp))/np.dot(Utemp, Utemp)
  if masked:
    rho_dot = MaskedArray(rho_dot >= 0, rho_dot)
  if plot:
    if vmin==vmax:
      lim = np.max(np.abs(rho_dot))
      vmin = -lim
      vmax = lim
    ax = plt.gca()
    mesh = ax.pcolormesh(X1, X2, rho_dot, cmap=cmap, vmin=vmin, vmax=vmax)
    clb = plt.colorbar(mesh)
    clb.ax.set_title('$\\dot{\\rho}$', fontsize=28, y=1.02)
    plt.xlim(xlims); plt.ylim(ylims)
    if savefig:
      plt.savefig(figname, transparent=True, bbox_inches='tight')
  if output:
    return x1, x2, rho_dot

def shear_rate(func, xlims, ylims, ds, *fargs, output=False, masked=False, plot=True, cmap='coolwarm', newfig=True, savefig=False, figname='localRho.pdf', vmin=None, vmax=None):
  if plot and newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      U[n, m], V[n, m] = func(y0, *fargs)
  [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
  q_dot = np.zeros(np.shape(U))
  R = np.array([[0, 1], [-1, 0]])
  for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      S = 0.5*(Grad + np.transpose(Grad))
      q_dot[n, m] = np.dot(Utemp, np.dot(np.dot(S, R) - np.dot(R, S), Utemp))/np.dot(Utemp, Utemp)
  if masked:
    q_dot = MaskedArray(q_dot >= 0, q_dot)
  if plot:
    if vmin==vmax:
      lim = np.max(np.abs(q_dot))
      vmin = -lim
      vmax = lim
    ax = plt.gca()
    mesh = ax.pcolormesh(X1, X2, q_dot, cmap=cmap, vmin=vmin, vmax=vmax)
    clb = plt.colorbar(mesh)
    clb.ax.set_title('$\\dot{q}$', fontsize=28, y=1.02)
    plt.xlim(xlims); plt.ylim(ylims)
    if savefig:
      plt.savefig(figname, transparent=True, bbox_inches='tight')
  if output:
    return x1, x2, q_dot

def stretch_rate(func, xlims, ylims, ds, *fargs, output=False, masked=False, plot=True, cmap='coolwarm', newfig=True, savefig=False, figname='localRho.pdf', vmin=None, vmax=None):
  if plot and newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      U[n, m], V[n, m] = func(y0, *fargs)
  [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
  p_dot = np.zeros(np.shape(U))
  R = np.array([[0, 1], [-1, 0]])
  for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      S = 0.5*(Grad + np.transpose(Grad))
      p_dot[n, m] = np.dot(Utemp, np.dot(S, Utemp))/np.dot(Utemp, Utemp)
  if masked:
    p_dot = MaskedArray(p_dot >= 0, p_dot)
  if plot:
    if vmin==vmax:
      lim = np.max(np.abs(p_dot))
      vmin = -lim
      vmax = lim
    ax = plt.gca()
    mesh = ax.pcolormesh(X1, X2, p_dot, cmap=cmap, vmin=vmin, vmax=vmax)
    clb = plt.colorbar(mesh)
    clb.ax.set_title('$\\dot{p}$', fontsize=28, y=1.02)
    plt.xlim(xlims); plt.ylim(ylims)
    if savefig:
      plt.savefig(figname, transparent=True, bbox_inches='tight')
  if output:
    return x1, x2, p_dot

def divergence_ratio(func, xlims, ylims, ds, output=False, plot=True, cmap='inferno', newfig=True, savefig=False, figname='localRho.pdf', vmin=None, vmax=None):
  """
  """
  if plot and newfig:
    goodfigure(xlims, ylims)
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  X1, X2 = np.meshgrid(x1, x2)
  U = np.zeros(np.shape(X1))
  V = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      y0 = np.array([X1[n, m], X2[n, m]])
      U[n, m], V[n, m] = func(y0)
  [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
  [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
  nu_dot = np.zeros(np.shape(U))
  for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
      S = 0.5*(Grad + np.transpose(Grad))
      Sigma = np.trace(S)*np.eye(2) - 2*S
      nu_dot[n, m] = np.dot(Utemp, np.dot(Sigma, Utemp))/np.dot(Utemp, Utemp)
  if plot:
    if vmin==vmax:
      lim = np.max(np.abs(nu_dot))
      vmin = -lim
      vmax = lim
    plt.pcolormesh(X1, X2, nu_dot, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlim(xlims); plt.ylim(ylims)
    if savefig:
      plt.savefig(figname, transparent=True, bbox_inches='tight')
  if output:
    return x1, x2, nu_dot

# def acceleration(func, xlims, ylims, ds, *fargs, output=False, masked=False, plot=True, cmap='coolwarm', newfig=True, savefig=False, figname='localRho.pdf', vmin=vmin, vmax=vmax):
#   """The trajectory-normal repulsion rate

#   This function finds the trajectory-normal repulsion rate field introduced by Nave and Ross, 2017.
#   Gives a measure of how much the trajectory passing through x_0 attracts or repels nearby
#   trajectories infinitesimally.

#   Parameters
#   ----------
#   func : function
#     A two-dimensional function f([x, y]) which returns [f1, f2]
  
#   xlims : length 2 list or tuple of floats
#     The x-axis limits of the figure
  
#   ylims : length 2 list or tuple of floats
#     The y-axis limits of the figure

#   ds : float
#     Grid spacing in both directions, assumed to be equal

#   *fargs : arguments to pass to func
  
#   output : boolean, optional, default: False
#     If set to True, returns outputs described below

#   plot : boolean, optional, default: True

#   cmap : colormap, optional, default: 'bone'
#     Selection of colormap from matplotlib.cmap

#   newfig : boolean, optional, default: True
#     Chooses whether phase_plot is plotted in a new figure. To put phase_plot
#     on top of an existing figure, set to False

#   savefig : boolean, optional, default: False
#     Chooses whether to save the figure as an image file, named with figname
#     Uses matplotlib.pyplot.savefig

#   figname : string, optional, default: 'repulsion_factor.png'
#     If savefig=True is used, specifies the name of the imagefile within the
#     matplotlib.pyplot.savefig command

#   Returns
#   -------
#   If plot=True, returns a matplotlib.pyplot.pcolormesh instance

#   If output=True, the following variables are returned:

#   x1 : 1-dimensional numpy.np.array
#     This represents the points along the x-axis

#   x2 : 1-dimensional numpy.np.array
#     This represents the points along the y-axis
#     To generate all points, use numpy.np.meshgrid(x1, x2)
  
#   rho_dot : scalar field
#     The trajectory-normal repulsion factor given by
#     rho_dot = <n_T, \nabla F^T n_0>


#   """
#   if plot and newfig:
#     goodfigure(xlims, ylims)
#   x1 = np.arange(xlims[0], xlims[1]+ds, ds)
#   x2 = np.arange(ylims[0], ylims[1]+ds, ds)
#   X1, X2 = np.meshgrid(x1, x2)
#   U = np.zeros(np.shape(X1))
#   V = np.zeros(np.shape(U))
#   for m in range(len(x1)):
#     for n in range(len(x2)):
#       y0 = np.array([X1[n, m], X2[n, m]])
#       U[n, m], V[n, m] = func(y0, *fargs)
#   [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
#   [DUxy, DUxx] = np.gradient(DUx[:, :], ds, edge_order=2)
#   [DUyy, DUyx] = np.gradient(DUy[:, :], ds, edge_order=2)
#   [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
#   [DVxy, DVxx] = np.gradient(DVx[:, :], ds, edge_order=2)
#   [DVyy, DVyx] = np.gradient(DVy[:, :], ds, edge_order=2)
#   rho_dot = np.zeros(np.shape(U))
#   J = np.array([[0, 1], [-1, 0]])
#   for m in range(len(x1)):
#     for n in range(len(x2)):
#       Utemp = np.array([U[n, m], V[n, m]])
#       Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
#       ddU = np.array([[DUxx[n, m], DUxy[n, m]], [DUyx[n, m], DUyy[n, m]]])
#       ddV = np.array([[DVxx[n, m], DVxy[n, m]], [DVyx[n, m], DVyy[n, m]]])
#       dA = 
#       S = 0.5*(Grad + np.transpose(Grad))
#       rho_dot[n, m] = np.dot(Utemp, np.dot(np.dot(np.transpose(J), np.dot(S, J)), Utemp))/np.dot(Utemp, Utemp)
#   if masked:
#     rho_dot = MaskedArray(rho_dot >= 0, rho_dot)
#   if plot:
#     lim = np.max(np.abs(rho_dot))
#     ax = plt.gca()
#     mesh = ax.pcolormesh(X1, X2, rho_dot, cmap=cmap, vmin=vmin, vmax=vmax)
#     clb = plt.colorbar(mesh)
#     clb.ax.set_title('$\\dot{\\rho}$', fontsize=28, y=1.02)
#     plt.xlim(xlims); plt.ylim(ylims)
#     if savefig:
#       plt.savefig(figname, transparent=True, bbox_inches='tight')
#   if output:
#     return x1, x2, rho_dot

def contour0(func, xlims, ylims, ds, *fargs, **kwargs):
  """The zero contour
  """
  #goodfigure(xlims, ylims)
  x1, x2, A = repulsion_rate(func, xlims, ylims, ds, *fargs, plot=False, newfig=False, output=True, **kwargs)
  X1, X2 = np.meshgrid(x1, x2)
  V1, V2 = func([X1, X2], *fargs)
  Ay, Ax = np.gradient(A, ds)
  plt.contour(X1, X2, (V2*Ax-V1*Ay)/(V1**2+V2**2), [0,], colors='r')

def ridge(func, xlims, ylims, ds, *fargs, **kwargs):
  """Ridge detection
  """
  x1, x2, A = repulsion_rate(func, xlims, ylims, ds, *fargs, plot=False, newfig=False, output=True, **kwargs)
  X1, X2 = np.meshgrid(x1, x2)
  V1, V2 = func([X1, X2], *fargs)
  Ay, Ax = np.gradient(A, ds)
  return x1, x2, (V2*Ax-V1*Ay)/(V1**2+V2**2)

def theorem_20(func, xlims, ylims, ds, *fargs, **kwargs):
  """
  """
  x1, x2, A = repulsion_rate(func, xlims, ylims, ds, *fargs, plot=True, newfig=True, output=True, **kwargs)
  x1, x2, V = repulsion_ratio_rate(func, xlims, ylims, ds, *fargs, plot=True, output=True, **kwargs)
  X1, X2 = np.meshgrid(x1, x2)
  V1, V2 = func([X1, X2], *fargs)
  Ay, Ax = np.gradient(A, ds)
  mask = np.ones(np.shape(Ax))
#  check = np.abs((V2*Ax-V1*Ay)/(V1**2+V2**2))
  for m in range(len(x1)):
    for n in range(len(x2)):
      if A[n, m] < 0 and V[n, m] < 0:
        mask[n, m] = 0
#  Am = np.ma.masked_where(np.abs((V2*Ax-V1*Ay)/(V1**2+V2**2)) < 0.01 and A< 0, A)
  cm.bone.set_bad('w', alpha=None)
  goodfigure(xlims, ylims)
  plt.pcolormesh(X1, X2, np.ma.np.array(A, mask=mask), cmap=cm.bone)
  plt.colorbar()
  #plt.gca().imshow(Am, interpolation='bilinear',
#                cmap='bone', origin='lower', extent = [xlims[0], xlims[1], ylims[0], ylims[1]])
  return np.ma.np.array(A, mask=mask)

# %% Example Functions
def kevrekidis(y, eps=0.01):
  return np.array([-y[0]-y[1]+2, 1/eps*(y[0]**3-y[1])])

def ex11(y):
  return -np.array([np.tanh(y[0]**2/4)+y[0],y[0]+2*y[1]])

def rotHoop(y, eps=0.1, gamma=2.3):
  return np.array([y[1], 1/eps*(np.sin(y[0])*(gamma*np.cos(y[0])-1)-y[1])])

# For eps=0.01 a=0.57662277221679688 works nicely
def vanderPol(y, eps=0.01, a=0.575):
  return np.array([1/eps*(y[1]-y[0]**3+y[0]), a-y[0]])

def Csquirrel(alpha):
  return 1, 1.5+2*np.cos(2*alpha)
def Cplate(alpha):
  return 1.4-np.cos(2*alpha), 1.2*np.sin(2*alpha)
def Cball(alpha):
  return np.array([2, 0])

def GliderAuto(v, theta=-np.pi/18, C=Cplate):
  #v = [vx, vz]
  psi = -np.arctan2(v[1], v[0])
  Yp = [(v[0]**2+v[1]**2)*(C(psi + theta)[1]*np.sin(psi)-C(psi + theta)[0]*np.cos(psi)), (v[0]**2+v[1]**2)*(C(psi + theta)[1]*np.cos(psi)+C(psi + theta)[0]*np.sin(psi))-1]
  return np.array(Yp)
