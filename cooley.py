# cooley.py
#
# Periodic sequences and discrete Fourier transform.

from math import sin, cos, pi
import numpy as np

# Periodic sequences.

class PeriodicSequence:
    def __init__(self, V):
        self._V = np.array(V, dtype=np.cdouble)
        self._N = self._V.shape[0]

    def sum(self):
        return np.sum(self._V)

    def average(self):
        return np.average(self._V)

    def norm(self):
        return np.linalg.norm(self._V)

    def norm_squared(self):
        return np.linalg.norm(self._V)**2
        
    def real(self):
        '''
        real() and realpart() would of course be the same,
        so we only keep one of them: real().

        '''
        return PeriodicSequence(np.real(self._V))

    def imagpart(self):
        '''
        Returns imaginary part according to mathematical convention,
        i.e., the real numbers that are multiplied by 1j to form
        the imaginary part.

        '''
        return PeriodicSequence(np.imag(self._V))

    def imag(self):
        '''
        Returns literal imaginary part, i.e., the imaginary 
        elements themselves.
        
        '''
        return PeriodicSequence(1j*np.imag(self._V))
    
    def conj(self):
        return PeriodicSequence(self._V.conj())

    def shift_right(self, k):
        return PeriodicSequence(np.roll(self._V, k)) 

    def shift_left(self, k):
        return PeriodicSequence(np.roll(self._V, -k)) 

    def reverse(self):
        return PSeq(self._V[::-1])

    def mirror(self, k=0):
        '''
        X = (X0 X1 X2 X3). Extend X to the left periodically,

        (X0 X1 X2 X3 X0 X1 X2 X3)

        Read off the mirror: 
        
        (X(-0) X(-1) X(-2) X(-3)) = (X0 X3 X2 X1).

        So in fact mirror is the same as reverse + right shift(1).

        We can extend definition of mirror.

        X.mirror(0) = X(-j)  ==> reverse X then right shift by 1
        X.mirror(1) = X(1-j) ==> reverse X then right shift by 2
        ...
        X.mirror(k) = X(k-j) ==> reverse X then right shift by k+1

        '''
        return self.reverse().shift_right(k + 1)

    def extent(self, K):
        '''
        X = (1 2 3)
        X.extent(6) = (1 2 3 1 2 3)
        X.extent(8) = (1 2 3 1 2 3 1 2)

        To do: implement for K < 0.

        '''
        return PSeq([self[i] for i in range(0,K)])  

    def stretch(self, K):
        stretched = PSeq([0]*self._N*K)
        for j in range(0, self._N):
            stretched._V[j*K] = self._V[j]
        return stretched

    def sample(self, K):
        '''
        Let K be a divisor of N. The sampled vector
        is of length N/K.

        '''
        sampled = PSeq([0]*(self._N//K))
        if not self._N % K == 0:
            raise ValueError('K={} does not divide N={}.'
                             .format(K, self._N))
        else:
            for j in range(0, self._N//K):
                sampled._V[j] = self._V[j*K]
        return sampled

    def convolution(self, other):
        '''
        self conv other via conv_matrix @ other_colvec.

        Compute Z = Y conv X.
        
            [Y0 Y2 Y1] [X0]
        Z = [Y1 Y0 Y2] [X1] = M.X
            [Y2 Y1 Y0] [X2]
    
        We can start at the bottom, with reversed Y, and shift_left
        as we go up each row. But there are ways to begin building
        M from the top row.

        Start with [Y0 Y1 Y2] and shift right as we move each row
        down. Then take the transpose to get M.

        [Y0 Y1 Y2]                 [Y0 Y2 Y1]
        [Y2 Y0 Y1] => transpose => [Y1 Y0 Y2] = M
        [Y1 Y2 Y0]                 [Y2 Y1 Y0]

        Another way. Note that top row is Y.mirror, which is reverse and then
        shift_right(1). Second row is reverse with shift_right(2) and 
        so on. Using the extended definition of mirror where

        Y.mirror(k) = Y.reverse.shift_right(k),

        We can elegantly build the convolution matrix by successively 
        mirroring Y to make the rows.

        '''
        if not len(self) == len(other):
            raise ValueError('Mismatch between PSeq of length {} '
                             'and PSeq of length {}.'
                             .format(self.N(), len(other)))
        else:
            M = np.array([self.mirror(k)._V for k in range(0, self._N)])
            return PSeq(M @ other._V)

    def lagged_product(self, other):
        '''
        Compute Z = X lagged Y.

            [X0 X1 X2] [Y0]
        Z = [X1 X2 X0] [Y1] 
            [X2 X0 X1] [Y2]

        Lagged product is not commutative. So in general

        X lagged Y   not =   Y lagged X.

        '''
        if not len(self) == len(other):
            raise ValueError('Mismatch between PSeq of length {} '
                             'and PSeq of length {}.'
                             .format(len(self), len(other)))
        else:
            M = np.array([self.shift_left(k)._V for k in range(0, self._N)])
            return PSeq(M @ other._V)
        
    def __len__(self):
        return len(self._V)

    def __add__(self, other):
        if isinstance(other, PSeq):
            return PeriodicSequence(self._V + other._V)
        else:
            return PSeq(self._V + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, PSeq):
            return PSeq(self._V - other._V)
        else:
            return PSeq(self._V - other)

    def __rsub__(self, other):
        return -self.__sub__(other) 

    def __mul__(self, other):
        if isinstance(other, PSeq):
            return PeriodicSequence(other._V * self._V)
        else:
            return PSeq(other * self._V)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, PSeq):
            return PSeq(self._V / other._V) 
        else:
            return PSeq(self._V / other)

    def __rtruediv__(self, other):
        ''' not implimented yet '''
        pass

    def __neg__(self):
        return PeriodicSequence(-self._V)
    
    def __eq__(self, other):
        return np.allclose(self._V, other._V)
    
    def __getitem__(self, key):
        return self._V[key % self._N]
    
    def __setitem__(self, key, value):
        self._V[key % self._N] = value
    
    def __repr__(self):
        return f'PSeq({str(self._V)})'

# Short version of the constructor.

PSeq = PeriodicSequence

# Predicates.

def is_pure_real(X): return X.conj() == X
def is_pure_imag(X): return X.conj() == -X
def is_even(X): return X.mirror() == X
def is_odd(X): return X.mirror() == -X

# Important periodic sequences.

def dirac_seq(N):
    z = [0]*N
    z[0] = 1
    return PSeq(z)

def ones_seq(N): return PSeq([1]*N)
def one_by_N_seq(N): return PSeq([1/N]*N)

# Utilities for creating examples and tests.

def random_periodic_seq(N):
    return PeriodicSequence(np.random.random(N) + 1j*np.random.random(N))



# TO DO:
# The next three should be constructed to be complex sequences and 
# not simply reals. 



def even_periodic_seq(N):
    return PeriodicSequence([cos(2*pi*k/N) for k in range(0,N)])

def another_even_periodic_seq(N):
    return PSeq([cos(4*pi*k/N) * cos(2*pi*k/N) for k in range(0,N)])

def odd_periodic_seq(N):
    return PeriodicSequence([sin(2*pi*k/N) for k in range(0,N)])

# Transforms.

class Transform:
    def __init__(self, N):
        self._N = N
        self._w = np.exp(2*pi*1.j/self._N)
        a = np.arange(0, self._N)
        b = np.arange(0, self._N).reshape(self._N, 1)
        self._W = self._w**(-(a*b))

    def ft(self, X):
        if not self._N == len(X):
            raise ValueError('Mismatch between transform  of length {} '
                    'and sequence of length {},'.format(self.N(), len(X)))
        return PeriodicSequence((self._W @ X._V) / self._N)
    
    def ift(self, A):
        if not self._N == len(A):
            raise ValueError('Mismatch between transform  of length {} '
                    'and sequence of length {}.'.format(self.N(), len(A)))
        return PeriodicSequence((self._W.conj().T) @ A._V)
    
    def __getitem__(self, tup):
        i, j = tup
        return self._W[i % self._N][ j % self._N]
    
    def N(self):
        return self._N
    
    def w(self):
        return self._w
    
    def W(self):
        return self._W
    
    def __repr__(self):
        return f'Transform({self._N})'

# the G function.

def G(N, K, j):
    '''
    The function called G in the Cooley paper.

    G(N, K, j) = 1 if j = 0 mod N.
    G(N, K, j) = (K/N) * (1 - w_K^j)/(1 - w_N^j) otherwise.

    We also have 

    G(N, K, J) = (K/N) sum_{n=0}^{N/K - 1} w_N^{n*j}.

    To do: 
        should we check if K divides N?

    '''
    wN = np.exp(2*pi*1j/N)
    wK = np.exp(2*pi*1j/K)
    if j % N == 0:
        return 1
    else:
        return (K/N) * (1 - wK**j) / (1 - wN**j)





