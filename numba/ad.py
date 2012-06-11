"""
Example of how to use byte-code execution technique to trace accesses to numpy arrays.

This file demonstrates two applications of this technique:
* optimize numpy computations for repeated calling
* provide automatic differentiation of procedural code

"""

import __builtin__
import os
import sys
import inspect
import trace
import opcode

import numpy as np
import theano

from .utils import itercode

from scipy.optimize.lbfgsb import fmin_l_bfgs_b

# Opcode help: http://docs.python.org/library/dis.html

# XXX: support full calling convention for named args, *args and **kwargs


# XXX: this is a crutch to the proof of concept, not meant to be part of the
# proposed API
class FrameVM(object):
    """
    A Class for evaluating a code block of CPython bytecode,
    and tracking accesses to numpy arrays.

    """
    def __init__(self, watcher, func):
        #print 'FrameVM', func
        self.watcher = watcher
        self.func = func
        self.fco = func.func_code
        self.names = self.fco.co_names
        self.varnames = self.fco.co_varnames
        self.constants = self.fco.co_consts
        self.costr = func.func_code.co_code
        self.argnames = self.fco.co_varnames[:self.fco.co_argcount]
        self.stack = []

    def call(self, args, kwargs):
        self.rval = None
        self._myglobals = {}
        for name in self.names:
            #print 'name', name
            try:
                self._myglobals[name] = self.func.func_globals[name]
            except KeyError:
                try:
                    self._myglobals[name] = __builtin__.__getattribute__(name)
                except AttributeError:
                    #print 'WARNING: name lookup failed', name
                    pass

        self._locals = [None] * len(self.fco.co_varnames)
        for i, name in enumerate(self.argnames):
            #print 'i', args, self.argnames, self.fco.co_varnames
            self._locals[i] = args[i]

        self.code_iter = itercode(self.costr)
        jmp = None
        while True:
            try:
                i, op, arg = self.code_iter.send(jmp)
            except StopIteration:
                break
            name = opcode.opname[op]
            # print 'OP: ', i, name
            jmp = getattr(self, 'op_' + name)(i, op, arg)

        return self.rval

    def op_BINARY_ADD(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 + arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars 
                or id(arg2) in self.watcher.svars):
            s1 = self.watcher.svars.get(id(arg1), arg1)
            s2 = self.watcher.svars.get(id(arg2), arg2)
            self.watcher.shadow(r, s1 + s2)
            #print 'added sym'

    def op_BINARY_SUBTRACT(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 - arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars 
                or id(arg2) in self.watcher.svars):
            s1 = self.watcher.svars.get(id(arg1), arg1)
            s2 = self.watcher.svars.get(id(arg2), arg2)
            self.watcher.shadow(r,  s1 - s2)

    def op_BINARY_MULTIPLY(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        r = arg1 * arg2
        self.stack.append(r)
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        if (id(arg1) in self.watcher.svars 
                or id(arg2) in self.watcher.svars):
            s1 = self.watcher.svars.get(id(arg1), arg1)
            s2 = self.watcher.svars.get(id(arg2), arg2)
            self.watcher.shadow(r, s1 * s2)
            #print 'mul sym', id(r)

    def op_BINARY_POWER(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        r = arg1 ** arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars 
                or id(arg2) in self.watcher.svars):
            s1 = self.watcher.svars.get(id(arg1), arg1)
            s2 = self.watcher.svars.get(id(arg2), arg2)
            self.watcher.shadow(r, s1 ** s2)
            #print 'mul sym', id(r)

    def op_CALL_FUNCTION(self, i, op, arg):
        # XXX: does this work with kwargs?
        # print 'CALL_FUNCTION', arg
        args = [self.stack[-ii] for ii in range(arg, 0, -1)]
        if arg > 0:
            self.stack = self.stack[:-arg]
        func = self.stack.pop(-1)
        recurse = True

        if (getattr(func, '__module__', None)
                and func.__module__.startswith('numpy')):
            recurse = False
        elif isinstance(func, np.ufunc):
            recurse = False

        if 'built-in' in str(func):
            recurse = False

        if recurse:
            # print 'stepping into', func
            vm = FrameVM(self.watcher, func)
            rval = vm.call(args, {})
        else:
            # print 'running built-in', func, func.__name__, args
            rval = func(*args)
            if any(id(a) in self.watcher.svars for a in args):
                sargs = [self.watcher.svars.get(id(a), a)
                        for a in args]
                if func.__name__ == 'sum':
                    self.watcher.shadow(rval, theano.tensor.sum(*sargs))
                elif func.__name__ == 'dot':
                    self.watcher.shadow(rval, theano.tensor.dot(*sargs))
                elif func.__name__ == 'mean':
                    self.watcher.shadow(rval, theano.tensor.mean(*sargs))
                elif func.__name__ == 'maximum':
                    self.watcher.shadow(rval, theano.tensor.maximum(*sargs))
                else:
                    raise NotImplementedError(func)
        self.stack.append(rval)

    def op_COMPARE_OP(self, i, op, arg):
        opname = opcode.cmp_op[arg]
        right = self.stack.pop(-1)
        left = self.stack.pop(-1)
        if 0: pass
        elif opname == '==': self.stack.append(left == right)
        elif opname == '!=': self.stack.append(left != right)
        elif opname == '>': self.stack.append(left > right)
        elif opname == '<': self.stack.append(left < right)
        else:
            raise NotImplementedError('comparison: %s' % opname)

        if any(id(a) in self.watcher.svars for a in [left, right]):
            sargs = [self.watcher.svars.get(id(a), a) for a in [left, right]]
            tos = self.stack[-1]
            if 0: pass
            elif opname == '<':
                self.watcher.shadow(tos, theano.tensor.lt(left, right))
            elif opname == '>':
                self.watcher.shadow(tos, theano.tensor.gt(left, right))
            else:
                raise NotImplementedError()

    def op_FOR_ITER(self, i, op, arg):
        # either push tos.next()
        # or pop tos and send (arg)
        tos = self.stack[-1]
        try:
            next = tos.next()
            # print 'next', next
            self.stack.append(next)
        except StopIteration:
            self.stack.pop(-1)
            return ('rel', arg)

    def op_JUMP_ABSOLUTE(self, i, op, arg):
        # print 'sending', arg
        return ('abs', arg)

    def op_JUMP_IF_TRUE(self, i, op, arg):
        tos = self.stack[-1]
        if tos:
            return ('rel', arg)

    def op_GET_ITER(self, i, op, arg):
        # replace tos -> iter(tos)
        tos = self.stack[-1]
        self.stack[-1] = iter(tos)
        if id(tos) in self.watcher.svars:
            raise NotImplementedError('iterator of watched value')

    def op_LOAD_GLOBAL(self, i, op, arg):
        # print 'LOAD_GLOBAL', self.names[arg]
        tos = self._myglobals[self.names[arg]]
        self.stack.append(tos)
        if (isinstance(tos, np.ndarray)
                and id(tos) not in self.watcher.svars):
            raise NotImplementedError()

    def op_LOAD_ATTR(self, i, op, arg):
        # print 'LOAD_ATTR', self.names[arg]
        TOS = self.stack[-1]
        self.stack[-1] = getattr(TOS, self.names[arg])
        tos = self.stack[-1]
        if (isinstance(tos, np.ndarray)
                and id(tos) not in self.watcher.svars):
            raise NotImplementedError()

    def op_LOAD_CONST(self, i, op, arg):
        # print 'LOAD_CONST', self.constants[arg]
        self.stack.append(self.constants[arg])
        tos = self.stack[-1]
        if (isinstance(tos, np.ndarray)
                and id(tos) not in self.watcher.svars):
            raise NotImplementedError()

    def op_LOAD_DEREF(self, i, op, arg):
        # print '???', i, op, arg
        # print self.func.func_closure
        thing = self.func.func_closure[arg]
        # print dir(thing.cell_contents)
        self.stack.append(thing.cell_contents)

    def op_LOAD_FAST(self, i, op, arg):
        # print 'LOAD_FAST', self.varnames[arg], self._locals[arg]
        tos = self._locals[arg]
        self.stack.append(tos)
        if (isinstance(tos, np.ndarray)
                and id(tos) not in self.watcher.svars):
            if tos.dtype == bool:
                print >> sys.stderr, "Warning: Theano has no bool, upgrading to uint8"
                s_tos = theano.shared(tos.astype('uint8'), borrow=False)
            else:
                s_tos = theano.shared(tos, borrow=False)
            self.watcher.shadow(tos, s_tos)

    def op_POP_BLOCK(self, i, op, arg):
        #print 'pop block, what to do?'
        pass

    def op_POP_TOP(self, i, op, arg):
        self.stack.pop(-1)

    def op_PRINT_ITEM(self, i, op, arg):
        print self.stack.pop(-1),

    def op_PRINT_NEWLINE(self, i, op, arg):
        print ''

    def op_SETUP_LOOP(self, i, op, arg):
        #print 'SETUP_LOOP, what to do?'
        pass

    def op_STORE_FAST(self, i, op, arg):
        #print 'STORE_FAST', self.varnames[arg], self.stack[-1]
        self._locals[arg] = self.stack.pop(-1)

    def op_RAISE_VARARGS(self, i, op, arg):
        if 1 <= arg:
            exc = self.stack.pop(-1)
        if 2 <= arg:
            param = self.stack.pop(-1)
        if 3 <= arg:
            tb = self.stack.pop(-1)
        raise NotImplementedError('exception handling')

    def op_RETURN_VALUE(self, i, op, arg):
        self.rval = self.stack.pop(-1)


# XXX: this is a crutch to the proof of concept, not meant to be part of the
# proposed API
class Context(object):
    def __init__(self):
        self.svars = {}
        self.nogc = [] # ids that must not be reused

    def shadow(self, rval, sval):
        self.svars[id(rval)] = sval
        self.nogc.append(rval)

    def call(self, fn, args=(), kwargs={}):
        vm = FrameVM(self, fn)
        return vm.call(args, kwargs)

    def grad_fn(self, rval, ival):
        sy = self.svars[id(rval)]
        sx = self.svars[id(ival)]
        dydx = theano.tensor.grad(sy, sx)
        return theano.function([sx], dydx)

    def recalculate_fn(self, rval, ival):
        sy = self.svars[id(rval)]
        sx = self.svars[id(ival)]
        return theano.function([sx], sy)


def gradient(fn, args_like=None):
    """
    Returns a function g(*args) that will compute:
        fn(*args), [gradient(x) for x in args]

    in which gradient(x) denotes the derivative in fn(args) wrt each argument.

    When `fn` returns a scalar then the gradients have the same shape as the
    arguments.  When `fn` returns a general ndarray, then the gradients
    have leading dimensions corresponding to the shape of the return value.

    fn - a function that returns a float or ndarray
    args_like - representative arguments, in terms of shape and dtype

    """
    # args must be all float or np.ndarray

    # inspect bytecode of fn to determine derivative wrt args

    # construct bytecode for f_df() that
    # * unpacks x-> args
    # * computes f, dx

    # unpack x_opt -> args-like quantity `args_opt`

def fmin(fn, args, algo=(fmin_l_bfgs_b, {})):
    """
    fn: a scalar-valued function of floats and float/complex arguments

    args: a list of floats / complex / ndarrays from from which to start
        optimizing `fn(*args)`

    algo: choose the optimization algorithm with (fmin, kwargs) tuple.
        In future: require cost to be differentiable with respect to all
        elements of `wrt` and optimize using fmin_l_bfgs_b

    """

    # STEP 1: inspect bytecode of fn to determine derivative wrt args

    # hacky way to get call graph (we could do it without actually running it)
    ctxt = Context()
    cost = ctxt.call(fn, args)


    # construct bytecode for f_df() that
    # * unpacks x-> args
    # * computes f, dx
    orig_s_args = [ctxt.svars[id(w)] for w in args]
    args_shapes = [w.shape for w in args]
    args_sizes = [w.size for w in args]
    x_size = sum(args_sizes)
    x = np.empty(x_size)
    s_x = theano.tensor.vector(dtype=x.dtype)
    s_args = []
    i = 0
    for w in args:
        x[i: i + w.size] = w.flatten()
        if w.shape:
            s_args.append(s_x[i: i + w.size].reshape(w.shape))
        else:
            s_args.append(s_x[i])
        i += w.size

    orig_s_cost = ctxt.svars[id(cost)]
    memo = theano.gof.graph.clone_get_equiv(
            theano.gof.graph.inputs([orig_s_cost]),
            [orig_s_cost],
            memo=dict(zip(orig_s_args, s_args)))
    s_cost = memo[orig_s_cost]
    g_x = theano.tensor.grad(s_cost, s_x)


    # [optional] pass bytecode for g() to numba.translate to compile a faster
    # implementation for the repeated calls that are coming up

    # XXX: hacky current thing does not pass by a proper byte-code optimizer
    # because numba isn't quite there yet. For now we just compile the call
    # graph we already built theano-style.
    f_df = theano.function([s_x], [s_cost, g_x])

    # pass control to iterative minimizer
    #x_opt, mincost, info_dct = fmin_l_bfgs_b(f_df, x, **fmin_kwargs)
    fmin, fmin_kwargs = algo
    x_opt, mincost, info_dct = fmin(f_df, x, **fmin_kwargs)

    # unpack x_opt -> args-like quantity `args_opt`
    rval = []
    i = 0
    for w in args:
        rval.append(x_opt[i: i + w.size].reshape(w.shape))
        i += w.size
    return rval #, mincost, info_dct
