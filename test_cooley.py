# test_pe.py
#
# These tests are all properties of the DFT 
# taken from the Cooley paper.

import unittest
from cooley import *

class TestSequences(unittest.TestCase):

    def test_shift(self):
        self.assertTrue((PSeq([0,1,2,3,4]).shift_right(3)
            == PSeq([2,3,4,0,1])))
        self.assertTrue((PSeq([0,1,2,3,4]).shift_left(3) 
            == PSeq([3,4,0,1,2])))

    def test_mirror(self):
        '''
        X(-j) must be mirror(X) which must be X(N-j).

        Mirror is equivalent to reverse + shift_right.

        '''
        N = 20
        X = random_periodic_seq(N);
        check = [X.mirror()[j]==X[N-j] for j in range(0,N)]
        self.assertTrue(all(check))
        self.assertTrue(X.mirror() == X.reverse().shift_right(1))

    def test_muldiv_arithmetic(self):
        '''
        Pointwise arithmetic with * and /.

        '''
        self.assertTrue(PSeq([1,2,3])*2 == PSeq([2,4,6])) 
        self.assertTrue(2*PSeq([1,2,3]) == PSeq([2,4,6])) 
        self.assertTrue(PSeq([1,2,3])*PSeq([10,20,30]) == PSeq([10,40,90]))
        self.assertTrue(PSeq([2,4,6])/2 == PSeq([1,2,3]))
        self.assertTrue(PSeq([4,9,16])/PSeq([2,3,4]) == PSeq([2,3,4]))

    def test_addsub_arithmetic(self):
        '''
        Pointwise arithmetic with + and -.

        '''
        self.assertTrue(PSeq([1,2,3])+2 == PSeq([3,4,5]))
        self.assertTrue(2+PSeq([1,2,3]) == PSeq([3,4,5]))
        self.assertTrue(PSeq([1,2,3])+PSeq([4,5,6]) == PSeq([5,7,9]))
        self.assertTrue(PSeq([1,2,3])-1 == PSeq([0,1,2]))
        self.assertTrue(1-PSeq([1,2,3]) == PSeq([0,-1,-2]))
        self.assertTrue(PSeq([1,2,3])-PSeq([3,2,1]) == PSeq([-2,0,2]))

    def test_reverse(self):
        '''
        Reverse of periodic sequence.

        '''
        N = 20
        X = random_periodic_seq(N)
        Xr = PSeq([X[N-k] for k in range(0, N)])
        self.assertTrue(X.mirror() == Xr)
        self.assertTrue(PSeq([1,2,3]).reverse() == PSeq([3,2,1]))

    def test_convolution_symmetry(self):
        '''
        Convolution is commutative.

        X conv Y must be Y conv X.

        '''
        N = 10
        X = random_periodic_seq(N)
        Y = random_periodic_seq(N)
        self.assertTrue(X.convolution(Y) == Y.convolution(X))

    def test_convolution_lagged_prod_relationship(self):
        '''
        X lag Y should be X.mirror conv Y.

        '''
        N = 20
        X = random_periodic_seq(N)
        Y = random_periodic_seq(N)
        self.assertTrue(X.lagged_product(Y) == X.convolution(Y.mirror()))

    def test_lagged_convolution_equality_condition(self):
        '''
        If Y real, even => X lagged Y == X conv Y.
        If X,Y real, even => X lagged Y == Y lagged X.

        '''
        N = 25
        X = random_periodic_seq(N)
        Y = even_periodic_seq(N)
        Z = another_even_periodic_seq(N)
        self.assertTrue(X.lagged_product(Y) == X.convolution(Y))
        self.assertTrue(Y.lagged_product(Z) == Z.lagged_product(Y))



class TestTransforms(unittest.TestCase):

    def test_symmetric_roots(self):
        '''
        Roots matrix must be symmetric and orthogonal.

        W is the matrix of roots of unity of order N, 
        then W is symmetric and W . adjoint(W) = NI.

        '''
        N = 20
        t = Transform(N)
        NI = N*np.eye(N)
        self.assertTrue(np.allclose(t.W(), t.W().T))
        self.assertTrue(np.allclose(t.W() @ t.W().conj().T, NI))

    def test_inverse(self):
        '''
        DFT and inverse composes to identity,

        We must have inverse_DFT(DFT(X)) = DFT(inverse_DFT(X)) = X.

        '''
        N = 20
        X = random_periodic_seq(N)        
        t = Transform(N)
        self.assertTrue(X == t.ift(t.ft(X)))
        self.assertTrue(X == t.ft(t.ift(X)))

    def test_linearity(self):
        '''
        DFT must be linear.
       
        Let a, b be complex constants. Then we must have:

        DFT(a*X + b*Y) = a*DFT(X) + b*DFT(Y).

        '''
        a = 2.0 + 3.0j
        b = 1.0 - 2.0j
        N = 20
        X = random_periodic_seq(N)
        Y = random_periodic_seq(N)
        t = Transform(N)
        self.assertTrue(t.ft(a*X + b*Y) == a*t.ft(X) + b*t.ft(Y))
            
    def test_mirror(self):
        '''
        X, A iff X(-j), A(-n).

        Let X, A be transform pairs and let 
        X(-j) = X(N-j), A(-n) = A(N-n) 
        be the "mirrors" of X, A.

        We must have X(-j), A(-n) as transform pairs also.
        Or, DFT(X)(-n) = DFT(X(-j)).
        
        '''
        N = 20
        X = random_periodic_seq(N)
        t = Transform(N)
        self.assertTrue(t.ft(X).mirror() == t.ft(X.mirror()))  

    def test_parity(self):
        '''
        X even iff A even, etc.

        Let X, A be a transform pair.
        Then X is even iff A is even 
        and X is odd iff A is odd.

        '''
        N = 20
        Xeven = even_periodic_seq(N)
        Xodd = odd_periodic_seq(N)
        t = Transform(N)
        self.assertTrue(is_even(t.ft(Xeven)))
        self.assertTrue(is_odd(t.ft(Xodd)))

    def test_conjugate_mirror(self):
        '''
        X, A => conj(X), conj(A)(-n) is a pair, etc.
        
        Let X, A be a transform pair.
        Then conj(X), conj(A)(-n) is a transform pair 
        and so is conj(X)(-j), conj(A).

        '''
        N = 20
        X = random_periodic_seq(N)
        t = Transform(N)
        A = t.ft(X)
        self.assertTrue(t.ft(X.conj()) == A.conj().mirror())
        self.assertTrue(t.ft(X.conj().mirror()) == A.conj())

    def test_real_imag_mirror(self):
        '''
        X real iff A = conj(A)(-n) etc.

        Let X, A be a transform pair.
        X is real iff A = conj(A)(-n).
        A is real iff X = conj(X)(-j).
        X is imaginary iff A = -conj(A)(-n).
        A is imaginary iff X = -conj(X)(-j).

        '''
        N = 20
        X = random_periodic_seq(N)
        A = random_periodic_seq(N)
        t = Transform(N)
        Xr = X.real()
        Xi = X.imag()
        Ar = A.real()
        Ai = A.imag()
        self.assertTrue(t.ft(Xr) == t.ft(Xr).conj().mirror())
        self.assertTrue(t.ift(Ar) == t.ift(Ar).conj().mirror())
        self.assertTrue(t.ft(Xi) == -t.ft(Xi).mirror().conj())
        self.assertTrue(t.ift(Ai) == -t.ift(Ai).conj().mirror())

    def test_real_parity(self):
        '''
        X is real and even iff A is real and even.
        X is real and odd iff A is imaginary and odd.

        '''
        N = 20
        Xre = even_periodic_seq(N).real()
        Xro = odd_periodic_seq(N).real()
        t = Transform(N)
        self.assertTrue(is_pure_real(t.ft(Xre)) and is_even(t.ft(Xre)))
        self.assertTrue(is_pure_imag(t.ft(Xro)) and is_odd(t.ft(Xro)))

    def test_imag_parity(self):
        '''
        X is imaginary and even iff A is imaginary and odd.
        X is imaginary and odd iff A is real and odd.

        '''
        N = 20
        Xie = even_periodic_seq(N).imag()
        Xio = odd_periodic_seq(N).imag()
        t = Transform(N)
        self.assertTrue(is_pure_imag(t.ft(Xie)) and is_odd(t.ft(Xie)))
        self.assertTrue(is_pure_real(t.ft(Xio)) and is_odd(t.ft(Xio)))

    def test_two_for_one(self):
        '''
        Can find DFT of two real seqs in one shot.

        X = X1 + i*X2, where X1, X2 are real. If X, A is
        transform pair with A = A1 + i*A2, then the transforms
        of X1 and X2 are given by:

        A1 = (A + conj(A)(-n)) / 2
        A2 = (A - conj(A)(-n)) / 2i

        '''
        N = 20
        t = Transform(N)
        X = random_periodic_seq(N)
        A = t.ft(X)
        A1 = (A + A.conj().mirror()) * (1/2)
        A2 = (A + -A.conj().mirror()) * (1/2j)
        self.assertTrue(t.ft(X.real()) == A1)
        self.assertTrue(t.ft(X.imagpart()) == A2)

    def test_time_shift(self):
        '''
        X, A pairs => X shift k, W[:,k].A pairs.

        '''
        N = 20
        k = 9
        t = Transform(N)
        X = random_periodic_seq(N)
        A = t.ft(X)
        self.assertTrue(t.ft(X.shift_right(k)) == PSeq(t.W()[:,k]) * A) 

    def test_frequency_shift(self):
        '''
        X, A pairs => W[m,:].conj X, A shift m are pairs.

        '''
        N = 20
        m = 11
        t = Transform(N)
        X = random_periodic_seq(N)
        A = t.ft(X)
        self.assertTrue(PSeq(t.W().conj()[m,:]) * X == t.ift(A.shift_right(m)))

    def test_zero_mean_sum(self):
        '''
        X(0) = sum(A), A(0) = average(A).

        '''
        N = 20
        t = Transform(N)
        X = random_periodic_seq(N)
        A = t.ft(X)
        self.assertTrue(np.isclose(X[0], A.sum()))
        self.assertTrue(np.isclose(A[0], X.average()))

    def test_product_convolution(self):
        '''
        X1,A1 and X1,A2 pairs => X1 conv X2, A1*A2 pairs.
        => X1*X2, A1 conv A2 are pairs.
        Note that * is pointwise multiplication.

        '''
        N = 20
        t = Transform(N)
        X1 = random_periodic_seq(N)
        X2 = random_periodic_seq(N)
        A1 = t.ft(X1)
        A2 = t.ft(X2)
        self.assertTrue((1/N) * t.ft(X1.convolution(X2)) == A1 * A2)
        self.assertTrue(t.ft(X1 * X2) == A1.convolution(A2))

    def test_mirror_lagged_product(self):
        '''
        X1, A1 and X2, A2 are transform pairs. 
        Then X1 lagged X2 and A1 conv mirror(A2) are pairs.

        '''
        N = 20
        t = Transform(N)
        X1 = random_periodic_seq(N)
        X2 = random_periodic_seq(N)
        A1 = t.ft(X1)
        A2 = t.ft(X2)
        self.assertTrue((1/N)*t.ft(X1.lagged_product(X2)) == A1 * A2.mirror())
        self.assertTrue((1/N)*t.ft(X2.lagged_product(X1)) == A1.mirror() * A2)
        self.assertTrue(t.ft(X1*X2.mirror()) == A1.lagged_product(A2))
        self.assertTrue(t.ft(X1.mirror()*X2) == A2.lagged_product(A1))

    def test_pareseval(self):
        '''
        X,A => (1/N) |X|**2 == |A|**2.

        '''
        N = 25
        t = Transform(N)
        X = random_periodic_seq(N)
        A = t.ft(X)
        self.assertTrue(np.isclose((1/N)*X.norm_squared(), A.norm_squared()))

    def test_stretch_relation(self):
        '''
        X,A pair => stretch(X, K), A/K is a pair
                 => X, stretch(A, K) is a pair.
        '''
        N = 25
        K = 13
        t = Transform(N)
        tt = Transform(N*K)
        X = random_periodic_seq(N)
        A = t.ft(X)
        self.assertTrue(tt.ft(X.stretch(K)) == (1/K)*A.extent(N*K))
        self.assertTrue(tt.ft(X.extent(N*K)) == A.stretch(K))





class TestSpecialSequences(unittest.TestCase):

    def test_dirac(self):
        '''
        Dirac(N), (1/N) is a transform pair.
        (1), Dirac is a transform pair.

        '''
        N = 20
        t = Transform(N)
        self.assertTrue(t.ft(dirac_seq(N)) == one_by_N_seq(N))
        self.assertTrue(t.ft(ones_seq(N)) == dirac_seq(N))

    def test_dirac_scaling(self):
        '''
        X, A pair => X+a, A+a*dirac is a pair.

        '''
        N = 20
        X = random_periodic_seq(N)
        a = 1+3j
        t = Transform(N)
        self.assertTrue(t.ft(X+a) == t.ft(X) + a*dirac_seq(N))
        self.assertTrue(t.ft(X-a) == t.ft(X) - a*dirac_seq(N))




if __name__ == '__main__': 
    unittest.main(verbosity=1)
    
    
