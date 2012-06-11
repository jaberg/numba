""" Sketch of implementation of some PaCal functions in terms of bytecode
stuff.

"""

def eval(fn, args=()):
    """
    Evaluate fn by tracking random variables rather than single draws from
    np.RandomState distributions.

    Arithmetic interactions between normal numbers and random variables
    become random Variables.

    XXX: How to account for joint
    Return the mean of random variable (s.) returned by fn
    """

    # trace through either one active path through eval, or *all* potential
    # code paths through eval and transform the bytecode of fn into bytecode
    # in which calls to np.random.* trigger the creation of RandomVariable
    # objects rather than the drawing of a particular random sample. 
    #
    # This transformation is very similar to what has to be done for automatic
    # differentiation, in which 

    raise NotImplementedError()


def mcmc(fn, args=(), kwargs={},
        given={},
        iter=10000, burn=5000, thin=2, rseed=1,
        verbose=False):
    """
    Some kind of generic MCMC algorithm (how does pymc pick what to do?)

    Evaluate fn(*args, **kwargs) by tracking random variables rather than
    single draws from np.RandomState distributions.  This function should
    return a dictionary, and the keys of that return value should match the
    keys of the `given` argument.
    
    `given` maps some of the keys returned by `fn` to fixed
    values, which conditions and creates a posterior distribution for the
    sampler.

    iter, burn, thin, rseed, verbose all configure the sampling algorithm.
    """

    # the structure of this function is somewhat analagous to ad.fmin()

    # values in `given` should probably be ndarray or numbers, although in the
    # case that a random choice has been used to index into a list of
    # constants, we might be able to work backward and figure out which one it
    # was.

    # inspect bytecode of `distribution` to establish the functional
    # relationship between the elements
    # the analysis here is similar to what eval has to do.

    # construct bytecode for the MCMC sampling algorithm g()
    # <I'm not sure what are the boundaries between off-the-shelf sampling
    # logic and the constructed bytecode.. maybe the entire inner loop of the
    # mcmc loop can be embedded into this bytecode?

    # [optional] pass bytecode for g() to numba.translate to compile a faster
    # implementation for the repeated calls that are coming up...

    # run the sampler for a while
    # collect the samples for the keys returned by `fn` that are not also
    # named in the `given` dictionary

    # return the collected samples

    raise NotImplementedError()

