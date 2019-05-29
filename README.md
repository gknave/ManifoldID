# ManifoldID

This Python package provides various methods for finding influential manifolds in the phase space of 2-dimensional ordinary differential equations (ODEs) that are time independent (extensions to time-dependent systems are currently underway). Some tools available in this package include:

 - The finite-time Lyapunov exponent, as presented in [Haller (2015)](http://www.georgehaller.com/reprints/annurev-fluid-010313-141322.pdf) and [Shadden et al. (2005)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.120.439&rep=rep1&type=pdf).
 - The trajectory divergence rate, as introduced in [Nave, Nolan, and Ross (2019)](https://garynave.files.wordpress.com/2019/05/nave-nolan-ross-2019.pdf)
 - The trajectory-normal repulsion rate, as introduced in [Haller (2010), Section 9](http://georgehaller.com/reprints/variLCS.pdf)

## Example Code
The code operates on functions that take in a two-dimensional vector <img src="https://latex.codecogs.com/svg.latex?y" /> and return a two-dimensional vector <img src="https://latex.codecogs.com/svg.latex?\dot{y}" />.

    def myFunction(y):
        ydot = <something>
        return ydot

Next, we can plot, for instance, the phase portrait using `phase_plot`

    import manifoldid as mid
    def duffing(y)
        ydot = [y[1] ,y[0]-y[0]**3]
    xlims = [-1.5, 1.5]
    ylims = [-1, 1]
    mid.phase_plot(duffing, xlims, ylims)

## Installation
`manifoldid` will soon be available on `pip`, but for now, you need to clone the github repository and run the setup.py file.

    git clone https://github.com/gknave/ManifoldID.git
    cd manifoldid
    python setup.py install

## To-do List
1. Parallelize!   
   Many methods in this package require the integration of many trajectories over a grid of initial conditions. Parallelizing the code would speed up performance and make it much more useable.

2. Extend to non-autonomous systems

3. Extensions to experimental data

## Contribute
If you would like to be a part of this project moving forward, there's lots to do! Just send me an email at [Gary.Nave@colorado.edu](mailto:Gary.Nave@colorado.edu), and we can talk about how to make `ManifoldID` even more awesome!

## License
MIT License

Copyright (c) 2018 Gary Nave

*This is research code. There will likely be bugs.*
