from warnings import warn
from functools import partial

import numpy as np
try: # If we don't have mpl, silently ignore it.
    import matplotlib as mpl
except ImportError:
    mpl = None
from scipy import sparse, ndimage

# A map from neuron type abbreviation to ordered list of parameters
# a, b, c, d, C, k, Vr, Vt from Dynamical Systems in Neuroscience.
# NB: many of these models have some extra bonus features in the book,
# used to more accurately reproduce traces from electrophysiological
# experiments in the appropriate model organisms. In particular,
#  - LTS caps the value of u but (along with a few other types) allows
#     it to influence the effective value of spike threshold and c.
#  - I can't implement FS yet because its u nullcline is nonlinear.
#  - Several other types have PWL u nullclines.
#  - Different cell types have different spike thresholds.
NEURON_TYPES = {
    'rs':  [0.03, -2, -50, 100, 100, 0.7, -60, -40],
    'ib':  [0.01,  5, -56, 130, 150, 1.2, -75, -45],
    'ch':  [0.03,  1, -40, 150,  50, 1.5, -60, -40],
    'lts': [0.03,  8, -53,  20, 100, 1.0, -56, -42],
    'ls':  [0.17,  5, -45, 100,  20, 0.3, -66, -40]}


class Organoid():
    """
    A simulated 2D culture of cortical cells using models from
    Dynamical Systems in Neuroscience, with synapses implemented as
    exponential PSPs for both excitatory and inhibitory cells.

    The model represents the excitability of a neuron using three
    phase variables: the membrane voltage v : mV, the "recovery" or
    "leakage" current u : pA, and the total synaptic input to each
    cell Isyn : pA.

    Additionally, the excitation model contains the following static
    parameters, on a per cell basis by providing arrays of size (N,):
     a : 1/ms time constant of recovery current
     b : nS steady-state conductance for recovery current
     c : mV membrane voltage after a downstroke
     d : pA bump to recovery current after a downstroke
     C : pF membrane capacitance
     k : nS/mV voltage-gated Na+ channel conductance
     Vr: mV resting membrane voltage when u=0
     Vt: mV threshold voltage when u=0
     Vp: mV action potential peak, after which reset happens

    The cells are assumed to be located at physical positions contained
    in the variable XY : um, but this is not used for anything other than
    display (simulated Ca2+ imaging etc) unless you use it to generate
    the weight matrix S or something.
    """
    def __init__(self, *args, XY, S, tau,
                 a, b, c, d,
                 C, k, Vr, Vt, Vp):

        self.N = S.shape[0]

        self.S = S
        self.XY = XY
        self.a, self.b, self.c, self.d = a, b, c, d
        self.C, self.k, = C, k
        self.Vr, self.Vt, self.Vp = Vr, Vt, Vp
        self.tau = tau
        self.VUIJ = np.zeros((4, self.N))
        self.reset()

    def reset(self):
        self.VUIJ[0,:] = self.Vr
        self.fired = self.V >= self.Vp
        self.VUIJ[1:,:] = 0

    def VUIJdot(self, Iin):
        NAcurrent = self.k*(self.V - self.Vr)*(self.V - self.Vt)
        Vdot = (NAcurrent - self.U + self.Isyn + Iin) / self.C
        Udot = self.a * (self.b*(self.V - self.Vr) - self.U)
        Idot = self.Jsyn / self.tau
        Jdot = -(self.Isyn + 2*self.Jsyn) / self.tau
        return np.array([Vdot, Udot, Idot, Jdot])

    def step(self, dt, Iin):
        """
        Update the state of the organoid by 1ms, and return the current
        organoid state and a boolean array indicating which cells fired.
        """

        # Apply the correction to any cells that crossed the AP peak
        # in the last update step, so that this step puts them into
        # the start of the refractory period.
        fired = self.fired
        self.V[fired] = self.c[fired]
        self.U[fired] += self.d[fired]

        # Note that we store the total synaptic input to each cell and
        # let that decay over time.  The reason is just so that this
        # has to happen only once per firing rather than once per
        # update.  The disadvantage is that the synaptic time constant
        # must be a global constant rather than per presynaptic cell
        # (per postsynaptic cell would be possible, but doesn't make
        # any sense).
        self.Jsyn += self.S[:,fired].sum(1) / self.tau

        # Actually do the stepping, using the midpoint method for
        # integration. This costs as much as halving the timestep
        # would in forward Euler, but increases the order to 2.
        k1 = self.VUIJdot(Iin)
        self.VUIJ += k1 * dt/2
        k2 = self.VUIJdot(Iin)
        self.VUIJ += k2*dt - k1*dt/2

        # Make a note of which cells this step has caused to fire,
        # then correct their membrane voltages down to the peak.  This
        # can make some of the traces look a little weird; it may be
        # prettier to adjust the previous point UP to self.Vp and set
        # this point to self.c, but that's not possible here since we
        # don't save all states.
        self.fired = self.V >= self.Vp
        self.V[self.fired] = self.Vp


    def initialize_stdp(self, tau=25, Aplus=0.005, Aminus=None):
        """
        Set the parameters for STDP. If you don't call this method, it
        automatically runs the first time you try to perform STDP,
        taking default parameter values from Song's 2000 paper.
        """ 
        if Aminus is None:
            Aminus = Aplus * 1.05

        self.stdp_tau = tau
        self.stdp_Aplus = Aplus
        self.stdp_Aminus = Aminus 

        self._stdp_trace = np.zeros_like(self.V)
            

    def stdp(self, dt):
        """
        Simulate spike-time dependent plasticity as an attempt at
        implementing Hebbian learning. The algorithm keeps a trace at
        each cell measuring how much it has been firing recently
        (typically this decays more slowly than synaptic current) 
        and uses this to measure causality.
        """
        try:
            self._stdp_trace
        except AttributeError:
            self.configure_stdp()

        # The old synaptic trace decays with time constant stdp_tau.
        self._stdp_trace *= np.exp(-dt / self.stdp_tau)

        # Cells which have positive STDP trace probably contributed to
        # the firing of any cells which are active now. Increase their
        # synaptic strength by a percentage proportional to the trace.
        self.S[fired,:] *= 1 + self.stdp_Aplus * self._stdp_trace

        # Cells which fired now probably didn't contribute much to the
        # STDP trace of other cells. Decrease the strength of those
        # synapses by a percentage proportional to the trace.
        self.S[:,fired] *= 1 - self.stdp_Aminus * self._stdp_trace

        # Add entries to the trace for current firings.
        self._stdp_trace[fired] += 1


    @property
    def V(self):
        return self.VUIJ[0,:]

    @V.setter
    def V(self, value):
        self.VUIJ[0,:] = value

    @property
    def U(self):
        return self.VUIJ[1,:]

    @U.setter
    def U(self, value):
        self.VUIJ[1,:] = value

    @property
    def Isyn(self):
        return self.VUIJ[2,:]

    @Isyn.setter
    def Isyn(self, value):
        self.VUIJ[2,:] = value

    @property
    def Jsyn(self):
        return self.VUIJ[3,:]

    @Jsyn.setter
    def Jsyn(self, value):
        self.VUIJ[3,:] = value


class ChargedMedium():
    """
    Models the evolution of charge distribution with time in a medium
    full of diffusing ionic charge carriers, making two assumptions:
     1) charges are interchangeable and have identical properties,
     2) and they do not interact, i.e. an electron gas model.
    """

    def __init__(self, Xgrid, Ygrid,
                 D=2, eps_rel=80):
        self.D = D
        self.X, self.Y = Xgrid, Ygrid
        self._grid = np.array(np.meshgrid(Xgrid, Ygrid, indexing='ij'))
        self.dx, self.dy = Xgrid[1] - Xgrid[0], Ygrid[1] - Ygrid[0]
        self.rho = np.zeros(Xgrid.shape + Ygrid.shape)
        self.sigma = np.sqrt(2*D) / np.array([self.dx, self.dy])

        # Somehow, this code is off by a factor of 10^6 unless
        # I make the units of the free space permittivity wrong
        # in the following way. (It should be 10^-6 times this.)
        self._eps_factor = 4*np.pi* eps_rel * 8.854187

    def immerse(self, organoid):
        "Immerse an organoid in this medium for measurement."
        self.org = organoid
        self.Vprev = self.org.V.copy()

        # Quauntize neurons to grid points. If any neurons are
        # outside the grid, an error will be thrown later...
        self._neuron_grid = np.array(
            [np.argmin(abs(self.org.XY[0] - self.X[:,None]), axis=0),
             np.argmin(abs(self.org.XY[1] - self.Y[:,None]), axis=0)])

    def step(self, dt):
        """
        Run one forward simulation step. This will not work if the
        size of the filter is too small!
        """

        # Take the change in medium charge density due to the
        # membrane currents of all cells. Implementation note:
        # coordinate-form sparse matrices sum the contributions
        # of duplicate coordinates, i.e. this has the semantics
        # of looping over the cell grid but does it in C rather
        # than Python for a bit of a speed boost. The equivalent
        # Python loop is commented out below in case anyone cares.
        charge = self.org.C*(self.Vprev - self.org.V) / (self.dx*self.dy)
        self.rho += sparse.coo_matrix((charge, self._neuron_grid),
                                      shape=self.rho.shape)

        # for i,(x,y) in enumerate(zip(*self._neuron_grid)):
        #     self.rho[x,y] += charge[i]

        # Take one timestep of the diffusion process. Raise a warning
        # if the timestep is too small for diffusion to appear.
        sigma = self.sigma * np.sqrt(dt)
        if np.any(4*sigma < 1):
            warn('Timestep too small or grid too coarse;'
                 ' diffusion step has no effect.',
                 RuntimeWarning)
        self.rho = ndimage.gaussian_filter(self.rho, mode='constant',
                                           sigma=sigma)

        # Copy a new previous voltage.
        self.Vprev = self.org.V.copy()

    def probe_at(self, points):
        "Set probe point locations."

        # Save the points for caching.
        self._points = points

        # Distance from the probe points to each grid point.
        self._r = np.linalg.norm(points[:,...,None,None] -
                                 self._grid[:,None,...], axis=0)

        # Distance from the probe points to each cell.
        self._d = np.linalg.norm(points[:,...,None] -
                                 self.org.XY[:,None,...], axis=0)

    def probe(self, points=None):
        """
        Probe the voltage at the currently selected probe points.
        Optionally, you can provide the points, and it will select
        them by calling probe_at() for you. The last value is cached
        since usually you'll probe the same point set repeatedly.
        """

        if points not in (None, self._points):
            self.probe_at(points)

        # Contribution from the medium charge distribution: the
        # integral of charge distribution divided by distance.
        dist = np.trapz(np.trapz(self.rho/self._r, self.Y), self.X)

        # Contribution from the charge trapped inside the cells:
        # the sum per cell of the difference from resting potential.
        cells = self.org.C * (1/self._d) @ (self.org.V - self.org.Vr)

        # Divide by 4 pi epsilon_0 to get the actual potential.
        return (dist + cells) / self._eps_factor


class Ca2tImage():
    """
    Generate a single image without taking ownership of the Organoid's
    simulation stepping. Each firing creates fluorescence which initially
    displays as a pixel of intensity REACTIVITY (with 1 being fully
    saturated) and decays exponentially from there at rate TAU.
    """
    def __init__(self, cell_position, cell_size,
                 tau, reactivity, fig=None, **kwargs):

        self.tau = tau
        self.X = np.zeros(cell_position.shape[1])

        # Create the scatter plot...
        self.fig = fig
        self.ax = self.fig.gca(aspect='equal')
        self.ax.patch.set_facecolor((0,0,0))
        self.scat = self.ax.scatter(*cell_position,
                                    s=cell_size, c=self.X,
                                    cmap='gray', alpha=0.5,
                                    norm=mpl.colors.Normalize(0,1),
                                    **kwargs)

    def animate(self, dt, events, **kwargs):
        func = partial(self.step, dt)
        frames = _gen_frame_events(dt, events)
        return mpl.animation.FuncAnimation(self.fig, func=func,
                                           interval=dt,
                                           frames=frames,
                                           **kwargs)

    def step(self, dt, events):
        """
        Given a timestep (which is used to determine the decay of the
        fluorescence level), plus the delta-t and cell index of each
        firing event since the last call, updates the fluorescence
        state and scatter plot.
        """
        self.X *= np.exp(-dt / self.tau)
        for time, cell in events:
            self.X[cell] += np.exp(-time / self.tau)

        self.scat.set_array(self.X)


def _gen_frame_events(dt, events):
    """
    Given a list of firing events in the form (time, cell index),
    groups them into batches of all events in time intervals of
    length dt.
    """
    T, evs = dt, []
    for time, cell in events:
        if time > T:
            yield evs
            T += dt
            evs = []
        evs.append((T - time, cell))


class ElectrodeArray():
    """
    An electrical microelectrode array: a rectangular grid where
    each point stimulates nearby cells in a Neurons object.

    You pass a specification of the grid geometry, then an amount
    of activation per pin (input should have the same shape as the
    grid), and the array becomes a callable that can be
    """
    def __init__(self, *args,
                 spacing=None, shape=None, dimensions=None,
                 points=None, offset=(0,0), radius=10,
                 activation):
        if points is None:
            if dimensions is None:
                px, py = np.mgrid[:shape[0], :shape[1]] * spacing
            elif shape is None:
                try: spacing[0]
                except TypeError as _:
                    spacing = [spacing, spacing]
                px, py = np.mgrid[:dimensions[0]:spacing[0],
                                  :dimensions[1]:spacing[1]]
            elif spacing is None:
                px, py = np.mgrid[:dimensions[0]:shape[0]*1j,
                                  :dimensions[1]:shape[1]*1j]
            px = px.flatten() # - px.mean()
            py = py.flatten() # - py.mean()
            points = np.array((px, py))

        self.points = points + np.asarray(offset).reshape((2,1))
        self.activation = activation
        self.radius = radius

    def insert(self, org, vr, alpha):
        """
        Insert this array into an organoid. Precomputes the connectivity
        matrix from the array's inputs to the cells. You need to provide
        the resting voltages vr and a per-cell scaling alpha as well.
        """
        # The distance from the ith cell to the jth probe.
        dij = org.XY.reshape((2,-1,1)) - self.points.reshape((2,1,-1))
        dij = np.linalg.norm(dij / self.radius, axis=0, ord=2)
        self.M = 1 / np.maximum(1, dij)
        self.M = np.diag(alpha) @ self.M
        self.org = org
        self.vr = vr

    def Vprobe(self):
        return self.M.T @ (self.vr - self.org.V)

    def Iout(self, t):
        return self.M @ self.activation(t)
