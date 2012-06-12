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

class Unassigned(object): """Unassigned value"""

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
        self.stack = []
        self._locals = None
        self._myglobals = None
        self.code_iter = None
        self.print_ops = False
        self.print_stack = False

        # self.varnames = self.fco.co_varnames
        # self.costr = func.func_code.co_code
        # self.argnames = self.fco.co_varnames[:self.fco.co_argcount]

    def call(self, args, kwargs):

        func = self.func
        func_code = self.func.func_code
        co_varnames = self.func.func_code.co_varnames
        co_argcount = self.func.func_code.co_argcount

        self._myglobals = {}
        self._locals = [Unassigned] * len(co_varnames)

        _locals = self._locals
        if hasattr(func, 'im_self'):
            _locals[0] = func.im_self
            bind_varnames = co_varnames[1:]
            bind_offset = 1
        else:
            bind_varnames = co_varnames
            bind_offset = 0

        for name in func_code.co_names:
            #print 'name', name
            try:
                self._myglobals[name] = func.func_globals[name]
            except KeyError:
                try:
                    self._myglobals[name] = __builtin__.__getattribute__(name)
                except AttributeError:
                    #print 'WARNING: name lookup failed', name
                    pass

        extra_args_ok = bool(func_code.co_flags & 0x04)
        extra_kwargs_ok = bool(func_code.co_flags & 0x08)

        # -- assert that my understanding of calling protocol is correct
        #
        # param_names: the sequence of function parameter names
        # args_param: [optional] the name of the *vargs parameter
        # kwargs_param: [optional] the name of the **kwargs parameter
        # pos_params: sequence of potentially-positional parameter names
        try:
            if extra_args_ok and extra_kwargs_ok:
                assert len(bind_varnames) >= co_argcount + 2
                param_names = bind_varnames[:co_argcount + 2]
                args_param = param_names[co_argcount]
                kwargs_param = param_names[co_argcount + 1]
                pos_params = param_names[:co_argcount]
            elif extra_kwargs_ok:
                assert len(bind_varnames) >= co_argcount + 1
                param_names = bind_varnames[:co_argcount + 1]
                kwargs_param = param_names[co_argcount]
                pos_params = param_names[:co_argcount]
            elif extra_args_ok:
                assert len(bind_varnames) >= co_argcount + 1
                param_names = bind_varnames[:co_argcount + 1]
                args_param = param_names[co_argcount]
                pos_params = param_names[:co_argcount]
            else:
                assert len(bind_varnames) >= co_argcount
                param_names = bind_varnames[:co_argcount]
                pos_params = param_names[:co_argcount]
        except AssertionError:
            print 'YIKES: MISUNDERSTANDING OF CALL PROTOCOL:',
            print co_argcount,
            print bind_varnames,
            print '%x' % func_code.co_flags
            raise

        if len(args) > co_argcount and not extra_args_ok:
            raise TypeError('Argument count exceeds number of positional params')

        # -- bind positional arguments
        for i, (param_i, arg_i) in enumerate(zip(param_names, args)):
            assert bind_varnames[i] == param_i
            _locals[i + bind_offset] = arg_i

        if extra_args_ok:
            _locals[co_varnames.index(args_param)] == args[co_argcount:]

        # -- bind keyword arguments
        if extra_kwargs_ok:
            kwargs_pos = co_varnames.index(kwargs_param)
            _locals[kwargs_pos] == {}

        for aname, aval in kwargs.items():
            try:
                pos = pos_params.index(aname) + bind_offset
            except ValueError:
                if extra_kwargs_ok:
                    _locals[kwargs_pos][aname] = aval
                    continue
                else:
                    raise TypeError('Unrecognized keyword argument', aname)
            if _locals[pos] == Unassigned:
                _locals[pos] = aval
            else:
                raise TypeError('Duplicate argument for parameter', aname)

        # -- find default values
        if func.func_defaults:
            defaults = func.func_defaults
            for ii, val in enumerate(defaults):
                if _locals[co_argcount - len(defaults) + ii] is Unassigned:
                   _locals[co_argcount - len(defaults) + ii] = val

        if 0:
            print 'BINDING'
            for name, lval in zip(co_varnames, _locals):
                print '  ', name, lval

        self.code_iter = itercode(func_code.co_code)
        jmp = None
        while not hasattr(self, 'rval'):
            try:
                i, op, arg = self.code_iter.send(jmp)
            except StopIteration:
                break
            name = opcode.opname[op]
            name = {
                    'SLICE+0': 'SLICE_PLUS_0',
                    'SLICE+1': 'SLICE_PLUS_1',
                    'SLICE+2': 'SLICE_PLUS_2',
                    'SLICE+3': 'SLICE_PLUS_3',
                    }.get(name, name)
            if self.print_ops:
                print 'OP: ', i, name
            if self.print_stack:
                print self.stack
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

    def op_BINARY_DIVIDE(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 / arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars 
                or id(arg2) in self.watcher.svars):
            s1 = self.watcher.svars.get(id(arg1), arg1)
            s2 = self.watcher.svars.get(id(arg2), arg2)
            self.watcher.shadow(r, s1 / s2)

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

    def op_BINARY_SUBSCR(self, i, op, arg):
        # Implements TOS = TOS1[TOS].
        tos1, tos = self.stack[-2:]
        #print 'tos', tos
        #print 'tos1', tos1
        rval = tos1[tos]
        self.stack[-2:] = [rval]
        w = self.watcher
        if id(tos) in w.svars or id(tos1) in w.svars:
            s_tos = w.svars.get(id(tos), tos)
            s_tos1 = w.svars.get(id(tos1), tos1)
            s_rval = s_tos1[s_tos]
            w.shadow(rval, s_rval)

    def op_BUILD_MAP(self, i, op, arg):
        self.stack.append({})

    def op_BUILD_SLICE(self, i, op, arg):
        if arg == 2:
            tos1, tos = self.stack[-2:]
            self.stack[-2:] = [slice(tos1, tos)]
        elif arg == 3:
            tos2, tos1, tos = self.stack[-3:]
            self.stack[-3:] = [slice(tos2, tos1, tos)]
        else:
            raise NotImplementedError()

    def op_BUILD_TUPLE(self, i, op, arg):
        if arg:
            t = tuple(self.stack[-arg:])
            self.stack[-arg:] = [t]
        else:
            self.stack.append(())

    def op_CALL_FUNCTION(self, i, op, arg):
        n_args = arg & 0xFF
        n_kwargs = (arg & 0xFF00) >> 8
        assert not (arg >> 16) # what would this stuff up here mean?
        kwargs = dict([(self.stack[-2 * ii], self.stack[-2 * ii + 1])
                for ii in range(n_kwargs, 0, -1)])
        args = [self.stack[-ii - 2 * n_kwargs] for ii in range(n_args, 0, -1)]
        if arg:
            self.stack = self.stack[:- n_args - 2 * n_kwargs]
        func = self.stack.pop(-1)
        recurse = True

        if (getattr(func, '__module__', None)
                and func.__module__.startswith('numpy')):
            recurse = False
        elif isinstance(func, np.ufunc):
            recurse = False

        if 'built-in' in str(func):
            recurse = False

        if hasattr(func, '__theano_op__'):
            rval = func(*args, **kwargs)
            all_args = args + kwargs.values()
            if any(id(a) in self.watcher.svars for a in all_args):
                sargs = [self.watcher.svars.get(id(a), a) for a in args]
                skwargs = dict([(kw, self.watcher.svars.get(id(val), val))
                    for kw, val in kwargs.items()])
                s_rval = func.__theano_op__(*sargs, **skwargs)
                self.watcher.shadow(rval, s_rval)
        elif recurse:
            print 'stepping into', func
            vm = FrameVM(self.watcher, func)
            rval = vm.call(args, kwargs)
        else:
            # print 'running built-in', func, func.__name__, args
            rval = func(*args, **kwargs)
            all_args = args + kwargs.values()
            if any(id(a) in self.watcher.svars for a in all_args):
                if kwargs:
                    raise NotImplementedError('kwargs and svars in %s' %
                            str(func))
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
                elif func.__name__ == 'zeros_like':
                    self.watcher.shadow(rval, theano.tensor.zeros_like(*sargs))
                elif func.__name__ == 'abs':
                    self.watcher.shadow(rval, abs(*sargs))
                elif func.__name__ == 'log10':
                    self.watcher.shadow(rval, theano.tensor.log10(*sargs))
                elif func.__name__ == 'setdefault':
                    # XXX verify that this is dict.setdefault
                    pass
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
        elif opname == 'is': self.stack.append(left is right)
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
        if id(tos) in self.watcher.svars:
            raise NotImplementedError('iterator of watched value')
        self.stack[-1] = iter(tos)

    def op_LOAD_GLOBAL(self, i, op, arg):
        # print 'LOAD_GLOBAL', self.names[arg]
        tos = self._myglobals[self.func.func_code.co_names[arg]]
        self.stack.append(tos)
        if (isinstance(tos, np.ndarray)
                and id(tos) not in self.watcher.svars):
            raise NotImplementedError()

    def op_LOAD_ATTR(self, i, op, arg):
        # print 'LOAD_ATTR', self.names[arg]
        TOS = self.stack[-1]
        self.stack[-1] = getattr(TOS, self.func.func_code.co_names[arg])
        tos = self.stack[-1]
        if (isinstance(tos, np.ndarray)
                and id(tos) not in self.watcher.svars):
            raise NotImplementedError()

    def op_LOAD_CONST(self, i, op, arg):
        self.stack.append(self.func.func_code.co_consts[arg])
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
        #print 'LOAD_FAST', self.func.func_code.co_varnames[arg], self._locals[arg]
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

    def op_POP_JUMP_IF_FALSE(self, i, op, arg):
        #tos = self.stack[-1]
        tos = self.stack.pop(-1)
        if not tos:
            return ('abs', arg)

    def op_POP_JUMP_IF_TRUE(self, i, op, arg):
        #tos = self.stack[-1]
        tos = self.stack.pop(-1)
        if tos:
            return ('abs', arg)

    def op_POP_TOP(self, i, op, arg):
        self.stack.pop(-1)

    def op_PRINT_ITEM(self, i, op, arg):
        thing = self.stack.pop(-1)
        if thing == 'PRINT_OPS:True':
            self.print_ops = True
        if thing == 'PRINT_STACK:True':
            self.print_stack = True
        print thing,

    def op_PRINT_NEWLINE(self, i, op, arg):
        print ''

    def op_SETUP_LOOP(self, i, op, arg):
        #print 'SETUP_LOOP, what to do?'
        pass

    def op_SLICE_PLUS_3(self, i, op, arg):
        # Implements TOS = TOS2[TOS1:TOS]
        TOS2, TOS1, TOS = self.stack[-3:]
        rval = TOS2[TOS1:TOS]
        self.stack[-3:] = [rval]

        watcher = self.watcher
        if any(id(t) in watcher.svars for t in [TOS, TOS1, TOS2]):
            s  = w.get(TOS)
            s1 = w.get(TOS1)
            s2 = w.get(TOS2)
            s_rval = s2[s1:s]
            self.watcher.shadow(rval, s_rval)


    def op_STORE_FAST(self, i, op, arg):
        #print 'STORE_FAST', self.varnames[arg], self.stack[-1]
        self._locals[arg] = self.stack.pop(-1)

    def op_STORE_MAP(self, i, op, arg):
        key = self.stack.pop(-1)
        val = self.stack.pop(-1)
        dct = self.stack[-1]
        dct[key] = val


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

    def op_ROT_TWO(self, i, op, arg):
        a = self.stack[-1]
        b = self.stack[-2]
        self.stack[-1] = b
        self.stack[-2] = a


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
    # XXX remove algo param, make each algo a separate fmin function

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


def fmin_sgd_iter(fn, args, stepsize):
    raise NotImplementedError()


def fmin_sgd(fn, args, stepsize):
    """
    """
    # XXX REFACTOR WITH FMIN

    # STEP 1: inspect bytecode of fn to determine derivative wrt args

    # hacky way to get call graph (we could do it without actually running it)
    ctxt = Context()
    cost = ctxt.call(fn, args)

    # construct bytecode for f_df() that
    # * unpacks x-> args
    # * computes f, dx
    s_args = [ctxt.svars[id(w)] for w in args]
    s_cost = ctxt.svars[id(cost)]

    g_args = theano.tensor.grad(s_cost, s_args)

    # [optional] pass bytecode for g() to numba.translate to compile a faster
    # implementation for the repeated calls that are coming up

    # XXX: hacky current thing does not pass by a proper byte-code optimizer
    # because numba isn't quite there yet. For now we just compile the call
    # graph we already built theano-style.
    update_fn = theano.function([], [s_cost],
            update=[(a, a - stepsize * g) for a, g, in zip(s_args, g_args)],
            )

    # XXX: stopping criterion here
    for i in xrange(10):
        print update_fn()

    return [a.get_value() for a in s_args]
