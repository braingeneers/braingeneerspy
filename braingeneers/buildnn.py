import numpy as np
import functools

class Neurons():
    """
    Base class for a culture of neurons. New neuronal models should
    only require modifying the constructor and the _step() method.
    """
    def __init__(self, N, dt):
        self.N = N
        self.input_synapses = []
        self.dt = dt
        self.reset()

    def reset(self):
        """
        Reset the states of all the neurons to their resting value.
        """
        self.fired = np.zeros(self.N, dtype=np.bool)
        for syn in self.input_synapses:
            syn.reset()

    def Isyn(self):
        """
        Compute the total input to this culture from all of its
        synaptic predecessors.
        """
        Isyn = np.zeros(self.N)
        for syn in self.input_synapses:
            Isyn += syn.output()
        return Isyn

    def list_firings(self, Iin, time):
        """
        Simulate the network for some amount of time. The return value
        is a list of pairs (time,index) indicating which cells fired
        when. These should be sorted by time.
        """
        events = []
        n_steps = int(np.ceil(time / self.dt))
        for step in range(n_steps):
            self.fired = self._step(Iin + self.Isyn())
            for idx in np.arange(self.N)[self.fired]:
                events.append((step*self.dt, idx))
            for syn in self.input_synapses:
                syn._step()
        return events

    def _step(self, Iin):
        """
        Simulate the neural culture forward one step.
        """
        raise NotImplementedError

    def spike_raster(self, Iin, time, bin_size):
        """
        Simulate the network for a fixed total time with a constant
        input parameter and return a spike raster: a matrix with
        dimensions (time, cell index) which contains True if a cell
        fired at that time.
        """
        n_bins = int(np.ceil(time / bin_size))
        raster = np.zeros((self.N, n_bins), dtype=np.bool)

        for t,i in self.list_firings(Iin, time):
            raster[i, int(t//bin_size)] = True
        return raster

    def total_firings(self, Iin, time):
        """
        Simulate the network for a fixed total time with a constant
        input parameter and return the total number of times each cell
        fired during that time.
        """
        counts = np.zeros(self.N, dtype=np.int)
        for _,i in self.list_firings(Iin, time):
            counts[i] += 1
        return counts

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt
        for syn in self.input_synapses:
            syn.dt = dt


class AggregateCulture(Neurons):
    """
    The basic disjoint union operation on neuronal cultures: collects
    multiple different groups of cells into an aggregate that can be
    simulated in an order-independent manner. Also handles the case
    where only the first `Nin` neurons in total are allowed to receive
    inputs by setting the rest of the inputs to zero.
    """
    def __init__(self, *cultures, Nin=None):
        self.cultures = cultures
        self.Nin = Nin

        N = sum(c.N for c in cultures)
        super().__init__(N, cultures[0].dt)

    def Isyn(self):
        Isyn_sub = np.hstack([c.Isyn() for c in self.cultures])
        return super().Isyn() + Isyn_sub

    def list_firings(self, Iin, time):
        if self.Nin is not None:
            Iin_partial = Iin
            Iin = np.zeros(self.N)
            Iin[:self.Nin] = Iin_partial
        return super().list_firings(Iin, time)

    def _step(self, Iin):
        # For each culture, update the corresponding part of the
        # firings array. I can't decide if this method is gnarly or
        # neat, but it does seem simpler than any way I could think of
        # where the slices or indices were generated in advance.
        idces = slice(0,0)
        for c in self.cultures:
            idces = slice(idces.start, idces.stop + c.N)
            c.fired = self.fired[idces] = c._step(Iin[idces])
            for syn in c.input_synapses:
                syn._step()
            idces = slice(idces.start + c.N, idces.stop)

        return self.fired

    def reset(self):
        for c in self.cultures:
            c.reset()
        super().reset()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt
        for syn in self.input_synapses:
            syn.dt = dt
        for c in self.cultures:
            c.dt = dt


class PoissonNeurons(Neurons):
    """
    Simple Poisson stochastic neurons: the input parameter is
    interpreted as the average firing rate of the cell. There is no
    refractory period, so the observed firing rate will decrease with
    increasing dt, only reaching the requested rate in the zero limit.
    """
    def _step(self, rates):
        return np.random.poisson(rates*self.dt, self.N) > 0


class RateCoding(PoissonNeurons):
    """
    Poisson neurons which rate-encode their inputs following a
    high-pass filter, a simple hybrid of spike rate and time coding
    schemes which allows fast responses to sudden events. (Step inputs
    will produce bursty output patterns, for example.)

    The parameter `alpha` determines how much of the lower frequencies
    are subtracted; the unit step response decays from 1 to 1-alpha.
    """
    def __init__(self, N, dt, tau, alpha=1.0):
        super().__init__(N,dt)
        self.tau = tau
        self.alpha = alpha

    def reset(self):
        self.rates = np.zeros(self.N)
        super().reset()

    def _step(self, rates):
        self.rates += self.dt/self.tau * (rates - self.rates)
        return super()._step(rates - self.alpha*self.rates)


class PlaceCells(PoissonNeurons):
    """
    A grid of Poisson neurons which rate-encode position within the
    Euclidean unit box in dimension `len(Nouts)`. If the input exactly
    matches a grid point, the output neuron corresponding to it will
    fire at a rate `a0`, falling off as a Gaussian with distance
    with with sigma set so that adjacent cells falling off half as fast
    in each dimension.
    """
    def __init__(self, Nouts, dt, a0):
        super().__init__(np.product(Nouts), dt)
        self.a0 = a0

        # Generate a sparse grid of N points with spacing 1/N in each
        # axis, offset evenly on both sides from the unit box.
        self.grid = np.ogrid[tuple(slice(1/N/2, 1-1/N/2, N*1j)
                                   for N in Nouts)]
        # Fit the "standard deviation" of our Gaussian to the
        # separation between points so points adjacent to an active
        # cell are firing at half the maximal rate. Actually this is
        # the standard deviation squared, multiplied by two, just
        # because it would be redundant to divide by 2 here.
        self.vars = [1/N**2 / np.log(2)
                     for N in Nouts]

    def _step(self, inputs):
        dxs = [(grid - inp)**2 / var
               for grid,var,inp in
               zip(self.grid, self.vars, inputs)]
        rates = self.a0 * np.exp(-sum(dxs)).flatten()
        return super()._step(rates)


class LIFNeurons(Neurons):
    """
    Leaky Integrate-and-Fire neuron model: the input value is
    interpreted as a rate of change in membrane voltage, which is
    integrated with an exponential leak (towards a resting value Vr)
    determined by the parameter tau. When V reaches Vp, it is reset
    automatically to Vr. Also, during the refractory time t_refrac
    after each firing, the cell does not respond to input.
    """
    def __init__(self, N, dt, *, Vr, Vp, tau, c=None, t_refrac=0):
        self.Vr = Vr
        self.c = np.ones(N) * (Vr if c is None else c)
        self.Vp = Vp
        self.tau = tau
        self.t_refrac = t_refrac
        super().__init__(N, dt)

    def reset(self):
        self.V = np.ones(self.N) * self.Vr
        self.timer = np.zeros(self.N)
        super().reset()

    def _step(self, Iin):
        # Do the resets AFTER the voltages have been returned because
        # plots etc will turn out nicer.
        self.V[self.fired] = self.c[self.fired]
        self.timer[self.fired] = self.t_refrac

        # Now integrate, using the midpoint method to make the
        # exponential work better maybe?
        dVdt = Iin - (self.V - self.Vr)/self.tau
        V_test = self.V + self.dt*dVdt
        dVdt = Iin - (V_test - self.Vr)/self.tau
        self.V += dVdt * self.dt

        # Detect firings and handle refractory period: new events
        # can't occur until the absolute refractory period is over,
        # but integration continues. This is a bit weird, but
        # consistent with the model of Diehl & Cook (2015).
        self.timer -= self.dt
        return (self.timer <= 0) & (self.V >= self.Vp)


class IzhikevichNeurons(Neurons):
    """
    The Izhikevich neuron model as presented in Dynamical Systems in
    Neuroscience (2003). In brief, it is an adaptive quadratic
    integrate-and-fire neuron, whose phase variables v and u represent
    the membrane voltage and a membrane leakage current.

    The individual neuron model takes the following parameters; the
    book provides values matching physiological cell types.
     a : 1/ms time constant of recovery current
     b : nS steady-state conductance for recovery current
     c : mV membrane voltage after a downstroke
     d : pA bump to recovery current after a downstroke
     C : pF membrane capacitance
     k : nS/mV voltage-gated Na+ channel conductance
     Vr: mV resting membrane voltage when u=0
     Vt: mV threshold voltage when u=0
     Vp: mV action potential peak, after which reset happens
    """

    # Define neuron types to make initialization simpler. These
    # parameter values are from Dynamical Systems in Neuroscience, but
    # some of the models aren't fully handled here: many allow the
    # value of `u` to affect spike peak voltage, and a few have only
    # piecewise linear differential equations for `u`.
    RS = {'a': 0.03, 'b': -2, 'c': -50, 'd': 100,
          'C': 100, 'k': 0.7, 'Vr': -60, 'Vt': -40, 'Vp': 35}
    IB = {'a': 0.01, 'b': 5, 'c': -56, 'd': 130,
          'C': 150, 'k': 1.2, 'Vr': -75, 'Vt': -45, 'Vp': 40}
    CH = {'a': 0.03, 'b': 1, 'c': -40, 'd': 150,
          'C': 50, 'k': 1.5, 'Vr': -60, 'Vt': -40, 'Vp': 25}
    LTS = {'a': 0.03, 'b': 8.0, 'c': -53, 'd': 20,
           'C': 100, 'k': 1.0, 'Vr': -56, 'Vt': -42, 'Vp': 20}
    LS = {'a': 0.17, 'b': 5, 'c': -45, 'd': 100,
          'C': 20, 'k': 0.3, 'Vr': -66, 'Vt': -40, 'Vp': 30}

    @staticmethod
    def cell_types(types, values):
        """
        Given a list of M cell type parameter dictionaries (of the
        same form as IzhikevichNeurons.RS) and an N×M coefficient
        matrix, creates a parameter dictionary for N neurons with
        parameters formed by linear combinations of the input
        parameter types.
        """
        ts = []
        for t in types:
            try:
                ts.append(getattr(IzhikevichNeurons, t))
            except TypeError:
                ts.append(t)
        return {
            p: values @ np.array([t[p] for t in ts])
            for p in 'a b c d C k Vr Vt Vp'.split()
        }

    def __init__(self, N, dt, *, a, b, c, d, C, k, Vr, Vt, Vp):
        self.a = a
        self.b = b
        self.c = c * np.ones(N)
        self.d = d * np.ones(N)
        self.C = C
        self.k = k
        self.Vr = Vr
        self.Vt = Vt
        self.Vp = Vp
        super().__init__(N, dt)

    def reset(self):
        self.VU = np.vstack((self.Vr * np.ones(self.N),
                             np.zeros(self.N)))
        super().reset()

    def _vudot(self, Iin):
        return self._vudot_at(Iin, self.VU)

    def _vudot_at(self, Iin, VU):
        VUdot = np.zeros((2, self.N))
        NAcurrent = self.k*(VU[0,:] - self.Vr)*(VU[0,:] - self.Vt)
        VUdot[0,:] = (NAcurrent - VU[1,:] + Iin) / self.C
        VUdot[1,:] = self.a * (self.b*(VU[0,:] - self.Vr) - VU[1,:])
        return VUdot

    def _step(self, Iin):
        self.V[self.fired] = self.c[self.fired]
        self.U[self.fired] += self.d[self.fired]

        VU_mid = self.VU + self._vudot(Iin)*self.dt/2
        self.VU += self._vudot_at(Iin, VU_mid)*self.dt

        return self.V >= self.Vp

    @property
    def V(self):
        return self.VU[0,:]
    @V.setter
    def V(self, V):
        self.VU[0,:] = V

    @property
    def U(self):
        return self.VU[1,:]
    @U.setter
    def U(self, U):
        self.VU[1,:] = U


class Synapses():
    """
    Base class for a group of synaptic connections between two neural
    cultures `inputs` and `outputs`.
    """
    def __init__(self, G, inputs, outputs=None):
        # Recurrent by default, but allow connections between two
        # separate groups of neurons.
        self.inputs = inputs
        if outputs is None:
            self.outputs = inputs
        else:
            self.outputs = outputs
            assert inputs.dt == outputs.dt, \
                'Synapses can only connect cultures with identical dt.'

        self.M = self.inputs.N
        self.N = self.outputs.N
        self.dt = self.inputs.dt

        # The synaptic matrix needs to be in the superclass so it can
        # be relied upon by mixins.
        self.G = np.asarray(G)
        assert self.G.shape == (self.N, self.M), \
            f'Synaptic matrix should be (N,M), but is {self.G.shape}'

        self.outputs.input_synapses.append(self)
        self.reset()

    def remove(self):
        """
        Removes this synapse group from the Neurons object to which it
        provides its output.
        """
        self.outputs.input_synapses.remove(self)

    def output(self):
        """
        Return the numerical input that these synapses should be
        providing to the postsynaptic cells, given the current state
        of the presynaptic cells.
        """
        raise NotImplementedError

    def _step(self):
        """
        If these synapses have intrinsic dynamics, advance them one
        timestep, returning nothing.
        """
        pass

    def reset(self):
        """
        Reset the state variables of the synapses to their resting
        values, to support resetting of neural cultures.
        """
        raise NotImplementedError


def SynapticScaling(self, *, tau, rate_target, Ascaling):
    """
    Modifies an existing synapse group object to add synaptic scaling
    by a biologically plausible purely local method common in the
    literature but with no single original source as far as I know.
    The neuron estimates its own firing rate with a slowly decaying
    synaptic trace, and all synaptic weights are linearly controlled
    with rate constant `Ascaling` towards the firing rate setpoint
    `rate_target`.
    """
    self.tau_scaling = tau
    self.rate_target = rate_target
    self.Ascaling = Ascaling

    @functools.wraps(self._step)
    def _step():
        _step.__wrapped__()

        # Continuous dynamics of the trace.
        self.x_scaling -= self.dt * self.x_scaling/self.tau_scaling

        # Control G towards a desired rate. This multiplication is a
        # first-order approximation to the exact integration of a
        # linear ODE, which has the same convergence characteristics
        # as the forward Euler used to update the firing traces, but
        # broadcasts so takes only O(N) space.
        x_err = self.x_scaling / self.rate_target - 1
        self.G *= 1 - self.Ascaling * x_err[:,np.newaxis] * self.dt

        # Update the trace to include postsynaptic firing events. The
        # scaling factor makes the trace an estimate of the cell's
        # actual firing rate.
        self.x_scaling[self.outputs.fired] += 1/self.tau_scaling
    self._step = _step

    @functools.wraps(self.reset)
    def reset():
        reset.__wrapped__()
        self.x_scaling = np.ones(self.N) * self.rate_target
    self.reset = reset

    self.reset()
    return self


def MinimalTripletSTDP(self, *, G_max=None,
                       tau_pre=16.8, tau_post1=33.7, tau_post2=125.,
                       Aplus=6.5e-3, Aminus=7.1e-3):
    """
    Modifies an existing synapse group object to add STDP by the
    triplet rule of Pfister and Gerstner (2006), using one presynaptic
    and two postsynaptic traces, all at different time constants.
    This is the minimal version where LTD doesn't include triplets
    and LTP doesn't include pairs, reducing the number of parameters
    to "only" five. The defaults are the ones Pfister & Gerstner
    originally fit to Sjöström's V1 experiments, from their Table 3.
    """
    # Save the time constants into the object so they act like normal
    # properties, e.g. can be modified by the user.
    self.tau_pre = tau_pre
    self.tau_post1 = tau_post1
    self.tau_post2 = tau_post2
    self.Aplus = Aplus
    self.Aminus = Aminus
    self.G_max = G_max

    # Add evolution of the traces to the step method.
    @functools.wraps(self._step)
    def _step():
        _step.__wrapped__()

        # First, update the continuous dynamics of the traces.
        self.x_pre -= self.x_pre * self.dt/self.tau_pre
        self.x_post1 -= self.x_post1 * self.dt/self.tau_post1
        self.x_post2 -= self.x_post2 * self.dt/self.tau_post2

        fo = self.outputs.fired
        if fo.any():
            # Synapses from cells which have fired recently onto cells
            # which just fired are subject to LTP.
            self.G[fo,:] += self.x_pre * \
                self.Aplus*self.x_post2[fo,np.newaxis]

            # Bump the synaptic input trace.
            self.x_post1[fo] += 1
            self.x_post2[fo] += 1

        fi = self.inputs.fired
        if fi.any():
            # Synapses from cells which haven't fired recently onto
            # cells which just fired are subject to LTD.
            self.G[:,fi] -= self.x_post1[:,np.newaxis]*self.Aminus

            # Make sure there's no way to get negative conductances.
            np.clip(self.G[:,fi], 0, self.G_max, out=self.G[:,fi])

            # Bump the synaptic input trace.
            self.x_pre[fi] += 1

    self._step = _step

    # Reset the traces too.
    @functools.wraps(self.reset)
    def reset():
        reset.__wrapped__()
        self.x_pre = np.zeros(self.M)
        self.x_post1 = np.zeros(self.N)
        self.x_post2 = np.zeros(self.N)
    self.reset = reset

    # Just like a regular Synapses.__init__ method, reset state.
    self.reset()
    return self


class ExponentialSynapses(Synapses):
    """
    A synaptic connection block where each presynaptic firing creates
    an exponentially-decaying synaptic conductance. To simplify
    things, the synaptic reversal potential is specified for an entire
    synapse group rather than per-presynaptic-neuron.

    Additionally, stochastic activity is supported by passing the
    parameters noise_event_rate and noise_event_size. These determine
    the frequency and magnitude of spontaneous synaptic activations.
    """
    def __init__(self, G, inputs, outputs=None, *,
                 tau, Vn, noise_rate=0):
        super().__init__(G, inputs, outputs)
        self.tau = tau
        self.Vn = Vn
        self.noise_rate = noise_rate

    def output(self):
        if self.noise_rate != 0:
            return (self.Vn - self.outputs.V) * (self.G @ self.a) + \
                self.G.sum(1)*self.noise_rate*np.random.randn(self.N)
        else:
            return (self.Vn - self.outputs.V) * (self.G @ self.a)

    def _step(self):
        self.a -= self.dt/self.tau * self.a
        self.a[self.inputs.fired] += 1

    def reset(self):
        self.a = np.zeros(self.M)
