import numpy as np

class Neurons():
    """
    Base class for a culture of neurons. New neuronal models should
    only require modifying the constructor and the _step() method.
    """
    def __init__(self, N, dt):
        self.N = N
        self.dt = dt
        self.fired = np.zeros(N, dtype=np.bool)
        self.input_synapses = []

    def Isyn(self):
        """
        Compute the total input to this culture from all of its
        synaptic predecessors.
        """
        Iin = np.zeros(self.N)
        for syn in self.input_synapses:
            Iin += syn.output()
        return Iin

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
        Simulate the neural culture for a short duration equal to the
        intrinsic dt of the model.
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

    def total_firings(self, total_time, Iin):
        """
        Simulate the network for a fixed total time with a constant
        input parameter and return the total number of times each cell
        fired during that time.
        """
        return self.spike_raster(Iin, total_time, self.dt).sum(1)


class AggregateCulture(Neurons):
    """
    The basic disjoint union operation on neuronal cultures: collects
    multiple different groups of cells into an aggregate that can be
    simulated in an order-independent manner.
    """
    def __init__(self, *cultures):
        self.cultures = cultures
        N = sum(c.N for c in cultures)

        dt = cultures[0].dt
        assert all(c.dt == dt for c in cultures)

        super().__init__(N, dt)

    def Isyn(self):
        return np.hstack([c.Isyn() for c in self.cultures])

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


class PoissonNeurons(Neurons):
    """
    Simple Poisson stochastic neurons: the input parameter is
    interpreted as the average firing rate of the cell. There is no
    refractory period, so the observed firing rate will decrease with
    increasing dt, only reaching the requested rate in the zero limit.
    """
    def _step(self, rates):
        return np.random.poisson(rates*self.dt, self.N) > 0


class LIFNeurons(Neurons):
    """
    Leaky Integrate-and-Fire neuron model: the input value is
    interpreted as a rate of change in membrane voltage, which is
    integrated with an exponential leak (towards a resting value Vr)
    determined by the parameter tau. When V reaches Vp, it is reset
    automatically to Vr. Also, the remaining refractory time is saved
    per cell; if this value is positive, the cell cannot fire.
    """
    def __init__(self, N, dt, *, Vr, Vp, tau, c=None, t_refrac=0):
        self.Vr = Vr
        self.c = np.ones(N) * (Vr if c is None else c)
        self.Vp = Vp
        self.tau = tau
        self.V = np.ones(N) * Vr
        self.timer = np.zeros(N)
        self.t_refrac = t_refrac
        super().__init__(N, dt)

    def _step(self, dVdt):
        # Do the resets AFTER the voltages have been returned because
        # plots etc will turn out nicer.
        self.V[self.fired] = self.c[self.fired]
        self.timer[self.fired] = self.t_refrac

        # Now integrate.
        dVdt = dVdt - (self.V - self.Vr)/self.tau
        self.V += dVdt * self.dt
        self.timer -= self.dt

        # And fire! :)
        return (self.V >= self.Vp) & (self.timer < 0)


class Synapses():
    """
    Base class for a group of synaptic connections between two neural
    cultures `inputs` and `outputs`.
    """
    def __init__(self, inputs, outputs=None):
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
        self.outputs.input_synapses.append(self)

    def output(self):
        """
        Return the numerical input that these synapses should be
        providing to the postsynaptic cells, given the current state
        of the presynaptic cells.
        """
        raise NotImplementedError

    def _step(self):
        """
        If these synapses have intrinsic dynamics, advance them by a
        time interval dt, returning nothing.
        """
        pass


class ExponentialSynapses(Synapses):
    """
    A synaptic connection block where each presynaptic firing creates
    an exponentially-decaying synaptic conductance. To simplify
    things, the synaptic reversal potential is specified for an entire
    synapse group rather than per-presynaptic-neuron.
    """
    def __init__(self, inputs, outputs=None, *, tau, G, Vn):
        super().__init__(inputs, outputs)
        self.G = np.asarray(G)
        assert self.G.shape == (self.N, self.M), \
            f'Synaptic matrix should be (N,M), but is {self.G.shape}'
        self.tau = tau
        self.Vn = Vn
        self.a = np.zeros(self.M)

    def output(self):
        return (self.G@self.a) * (self.Vn - self.outputs.V)

    def _step(self):
        self.a[self.inputs.fired] += 1
        self.a -= self.dt/self.tau * self.a


class DiehlCook2015(AggregateCulture):
    def __init__(self, N, dt, *, Vr, Vp, tau_exc, tau_inh, tau_syn):
        self.input_layer = PoissonNeurons(784, dt)
        self.exc = LIFNeurons(N, dt, Vr=Vr, Vp=Vp, tau=tau_exc)
        self.inh = LIFNeurons(N, dt, Vr=Vr, Vp=Vp, tau=tau_inh)

        ExponentialSynapses(self.input_layer, self.exc,
                            tau=tau_syn, Vn=Vp,
                            G=np.random.rand(N,784))
        ExponentialSynapses(self.exc, self.inh,
                            tau=tau_syn, Vn=Vp,
                            G=np.eye(N))
        ExponentialSynapses(self.inh, self.exc,
                            tau=tau_syn, Vn=Vr,
                            G=1-np.eye(N))

        super().__init__(self.input_layer, self.exc, self.inh)

    def present(self, digit, off_time=150, on_time=350):
        """
        Present a digit to the network: first, allow the network to
        relax with zero input, then use the input digit as the rate
        argument to the input layer, and return the results.
        """
        self.total_firings(off_time, np.zeros(self.N))

        # Flatten the input digit and provide it to the input layer,
        # but also include zeros to send to the rest of the neurons.
        Iin = np.zeros(self.N)
        Iin[:len(digit.ravel())] = digit.flatten()
        return self.total_firings(on_time, Iin)
