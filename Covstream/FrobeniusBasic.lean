import Covstream.Welford

/-!
`Covstream.FrobeniusBasic` contains the matrix algebra used by the optimization
layer.

The most important theorem in this file is
`frobeniusNormSq_add_smul_eq_quadratic`, which turns the squared Frobenius norm
of `E + α D` into an explicit quadratic polynomial in `α`.
-/

namespace Covstream

section FrobeniusBasic

/-- Pointwise matrix addition in the coordinate representation. -/
def matrixAdd {k : Nat} (A B : CovMatrix k) : CovMatrix k :=
  fun i j => A i j + B i j

/-- Pointwise matrix subtraction in the coordinate representation. -/
def matrixSub {k : Nat} (A B : CovMatrix k) : CovMatrix k :=
  fun i j => A i j - B i j

/-- Scalar multiplication in the coordinate representation. -/
def matrixSmul {k : Nat} (α : Real) (A : CovMatrix k) : CovMatrix k :=
  fun i j => α * A i j

/-- Frobenius inner product on coordinatewise matrices. -/
def frobeniusInner {k : Nat} (A B : CovMatrix k) : Real :=
  ∑ i : Fin k, ∑ j : Fin k, A i j * B i j

/-- Squared Frobenius norm. -/
def frobeniusNormSq {k : Nat} (A : CovMatrix k) : Real :=
  frobeniusInner A A

theorem frobeniusNormSq_nonneg {k : Nat}
    (A : CovMatrix k) :
    0 ≤ frobeniusNormSq A := by
  unfold frobeniusNormSq frobeniusInner
  apply Finset.sum_nonneg
  intro i _
  apply Finset.sum_nonneg
  intro j _
  simpa [pow_two] using sq_nonneg (A i j)

section InternalLinearAlgebra

theorem frobeniusInner_add_left {k : Nat}
    (A B C : CovMatrix k) :
    frobeniusInner (matrixAdd A B) C = frobeniusInner A C + frobeniusInner B C := by
  calc
    frobeniusInner (matrixAdd A B) C
        = ∑ i : Fin k, ∑ j : Fin k, (A i j * C i j + B i j * C i j) := by
            unfold frobeniusInner matrixAdd
            apply Finset.sum_congr rfl
            intro i _
            apply Finset.sum_congr rfl
            intro j _
            ring
    _ = ∑ i : Fin k, ((∑ j : Fin k, A i j * C i j) + ∑ j : Fin k, B i j * C i j) := by
            apply Finset.sum_congr rfl
            intro i _
            rw [Finset.sum_add_distrib]
    _ = frobeniusInner A C + frobeniusInner B C := by
            unfold frobeniusInner
            rw [Finset.sum_add_distrib]

theorem frobeniusInner_smul_left {k : Nat}
    (α : Real) (A B : CovMatrix k) :
    frobeniusInner (matrixSmul α A) B = α * frobeniusInner A B := by
  calc
    frobeniusInner (matrixSmul α A) B
        = ∑ i : Fin k, ∑ j : Fin k, α * (A i j * B i j) := by
            unfold frobeniusInner matrixSmul
            apply Finset.sum_congr rfl
            intro i _
            apply Finset.sum_congr rfl
            intro j _
            ring
    _ = ∑ i : Fin k, α * ∑ j : Fin k, A i j * B i j := by
            apply Finset.sum_congr rfl
            intro i _
            rw [Finset.mul_sum]
    _ = α * frobeniusInner A B := by
            unfold frobeniusInner
            rw [Finset.mul_sum]

theorem frobeniusInner_comm {k : Nat}
    (A B : CovMatrix k) :
    frobeniusInner A B = frobeniusInner B A := by
  unfold frobeniusInner
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem frobeniusInner_add_right {k : Nat}
    (A B C : CovMatrix k) :
    frobeniusInner A (matrixAdd B C) = frobeniusInner A B + frobeniusInner A C := by
  rw [frobeniusInner_comm, frobeniusInner_add_left, frobeniusInner_comm B A, frobeniusInner_comm C A]

theorem frobeniusInner_smul_right {k : Nat}
    (α : Real) (A B : CovMatrix k) :
    frobeniusInner A (matrixSmul α B) = α * frobeniusInner A B := by
  rw [frobeniusInner_comm, frobeniusInner_smul_left, frobeniusInner_comm B A]

end InternalLinearAlgebra

/--
Quadratic expansion of the squared Frobenius norm.

This is the basic algebraic identity used in the optimization file.
-/
theorem frobeniusNormSq_add_smul_eq_quadratic {k : Nat}
    (E D : CovMatrix k) (α : Real) :
    frobeniusNormSq (matrixAdd E (matrixSmul α D))
      = frobeniusNormSq E + 2 * α * frobeniusInner E D + α ^ 2 * frobeniusNormSq D := by
  calc
    frobeniusNormSq (matrixAdd E (matrixSmul α D))
        = frobeniusInner (matrixAdd E (matrixSmul α D)) (matrixAdd E (matrixSmul α D)) := by
            rfl
    _ = frobeniusInner E (matrixAdd E (matrixSmul α D))
          + frobeniusInner (matrixSmul α D) (matrixAdd E (matrixSmul α D)) := by
            rw [frobeniusInner_add_left]
    _ = (frobeniusInner E E + frobeniusInner E (matrixSmul α D))
          + (frobeniusInner (matrixSmul α D) E + frobeniusInner (matrixSmul α D) (matrixSmul α D)) := by
            rw [frobeniusInner_add_right, frobeniusInner_add_right]
    _ = (frobeniusNormSq E + α * frobeniusInner E D)
          + (α * frobeniusInner D E + α * (α * frobeniusNormSq D)) := by
            rw [frobeniusNormSq, frobeniusInner_smul_right, frobeniusInner_smul_left,
              frobeniusInner_smul_left, frobeniusInner_smul_right, frobeniusNormSq]
    _ = frobeniusNormSq E + 2 * α * frobeniusInner E D + α ^ 2 * frobeniusNormSq D := by
            rw [frobeniusInner_comm D E]
            ring

end FrobeniusBasic

end Covstream
