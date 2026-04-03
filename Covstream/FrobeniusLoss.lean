import Covstream.LedoitWolf

/-!
`Covstream.FrobeniusLoss` contains the optimization layer behind Ledoit-Wolf.

It defines:

1. Basic matrix arithmetic in the function representation.
2. Frobenius inner product and squared Frobenius norm.
3. The shrinkage loss as a quadratic in the shrinkage coefficient.
4. The unconstrained optimizer for that quadratic objective.
-/

namespace Covstream

section FrobeniusLoss

/-- Pointwise matrix addition in the function representation. -/
def matrixAdd {k : Nat} (A B : CovMatrix k) : CovMatrix k :=
  fun i j => A i j + B i j

def matrixSub {k : Nat} (A B : CovMatrix k) : CovMatrix k :=
  fun i j => A i j - B i j

def matrixSmul {k : Nat} (α : Real) (A : CovMatrix k) : CovMatrix k :=
  fun i j => α * A i j

def frobeniusInner {k : Nat} (A B : CovMatrix k) : Real :=
  ∑ i : Fin k, ∑ j : Fin k, A i j * B i j

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

theorem frobeniusNormSq_add_smul {k : Nat}
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

def shrinkageResidual {k : Nat}
    (α : Real) (S T Sigma : CovMatrix k) : CovMatrix k :=
  matrixSub (shrinkMatrix α S T) Sigma

def shrinkageDirection {k : Nat}
    (S T : CovMatrix k) : CovMatrix k :=
  matrixSub T S

def estimationError {k : Nat}
    (S Sigma : CovMatrix k) : CovMatrix k :=
  matrixSub S Sigma

def shrinkageLoss {k : Nat}
    (α : Real) (S T Sigma : CovMatrix k) : Real :=
  frobeniusNormSq (shrinkageResidual α S T Sigma)

noncomputable def rawShrinkageCoeff {k : Nat}
    (S T Sigma : CovMatrix k) : Real :=
  - frobeniusInner (estimationError S Sigma) (shrinkageDirection S T)
    / frobeniusNormSq (shrinkageDirection S T)

theorem shrinkageResidual_eq {k : Nat}
    (α : Real) (S T Sigma : CovMatrix k) :
    shrinkageResidual α S T Sigma
      = matrixAdd (estimationError S Sigma) (matrixSmul α (shrinkageDirection S T)) := by
  funext i j
  simp [shrinkageResidual, estimationError, shrinkageDirection, matrixAdd, matrixSub, matrixSmul, shrinkMatrix]
  ring

theorem shrinkageLoss_quadratic {k : Nat}
    (α : Real) (S T Sigma : CovMatrix k) :
    shrinkageLoss α S T Sigma
      = frobeniusNormSq (estimationError S Sigma)
        + 2 * α * frobeniusInner (estimationError S Sigma) (shrinkageDirection S T)
        + α ^ 2 * frobeniusNormSq (shrinkageDirection S T) := by
  unfold shrinkageLoss
  rw [shrinkageResidual_eq]
  simpa using (frobeniusNormSq_add_smul (estimationError S Sigma) (shrinkageDirection S T) α)

theorem rawShrinkageCoeff_optimal {k : Nat}
    (S T Sigma : CovMatrix k)
    (hdir : frobeniusNormSq (shrinkageDirection S T) ≠ 0)
    (α : Real) :
    shrinkageLoss (rawShrinkageCoeff S T Sigma) S T Sigma ≤ shrinkageLoss α S T Sigma := by
  have hDnonneg : 0 ≤ frobeniusNormSq (shrinkageDirection S T) :=
    frobeniusNormSq_nonneg (shrinkageDirection S T)
  rw [shrinkageLoss_quadratic, shrinkageLoss_quadratic]
  let a := frobeniusNormSq (shrinkageDirection S T)
  let b := frobeniusInner (estimationError S Sigma) (shrinkageDirection S T)
  have ha : 0 < a := by
    have hane : a ≠ 0 := by
      simpa [a] using hdir
    have hage : 0 ≤ a := by
      simpa [a] using hDnonneg
    exact lt_of_le_of_ne hage (by simpa using hane.symm)
  have hraw : rawShrinkageCoeff S T Sigma = -b / a := by
    simp [rawShrinkageCoeff, a, b]
  rw [hraw]
  have ha_ne : a ≠ 0 := ne_of_gt ha
  have hleft : 2 * (-b / a) * b + (-b / a) ^ 2 * a = -b ^ 2 / a := by
    field_simp [ha_ne]
    ring
  have hcore : -b ^ 2 / a ≤ 2 * α * b + α ^ 2 * a := by
    rw [div_le_iff₀ ha]
    nlinarith [sq_nonneg (a * α + b)]
  have hgoal :
      frobeniusNormSq (estimationError S Sigma) + 2 * (-b / a) * b + (-b / a) ^ 2 * a
        ≤ frobeniusNormSq (estimationError S Sigma) + 2 * α * b + α ^ 2 * a := by
    calc
      frobeniusNormSq (estimationError S Sigma) + 2 * (-b / a) * b + (-b / a) ^ 2 * a
          = frobeniusNormSq (estimationError S Sigma) + (-b ^ 2 / a) := by
              simpa [add_assoc] using
                congrArg (fun t => frobeniusNormSq (estimationError S Sigma) + t) hleft
      _ ≤ frobeniusNormSq (estimationError S Sigma) + (2 * α * b + α ^ 2 * a) := by
          nlinarith [hcore]
      _ = frobeniusNormSq (estimationError S Sigma) + 2 * α * b + α ^ 2 * a := by ring
  simpa [a, b] using hgoal

/-- Clamp the unconstrained Frobenius optimum into the valid shrinkage interval. -/
noncomputable def clippedRawShrinkageCoeff {k : Nat}
    (S T Sigma : CovMatrix k) : Real :=
  clip01 (rawShrinkageCoeff S T Sigma)

theorem clippedRawShrinkageCoeff_nonneg {k : Nat}
    (S T Sigma : CovMatrix k) :
    0 ≤ clippedRawShrinkageCoeff S T Sigma := by
  unfold clippedRawShrinkageCoeff
  exact clip01_nonneg _

theorem clippedRawShrinkageCoeff_le_one {k : Nat}
    (S T Sigma : CovMatrix k) :
    clippedRawShrinkageCoeff S T Sigma ≤ 1 := by
  unfold clippedRawShrinkageCoeff
  exact clip01_le_one _

/-- Oracle shrinkage uses the clipped Frobenius-optimal coefficient. -/
noncomputable def oracleLedoitWolfShrink {k : Nat}
    (S T Sigma : CovMatrix k) : CovMatrix k :=
  shrinkMatrix (clippedRawShrinkageCoeff S T Sigma) S T

theorem oracleLedoitWolfShrink_symm {k : Nat}
    (S T Sigma : CovMatrix k)
    (hS : Symmetric S) (hT : Symmetric T) :
    Symmetric (oracleLedoitWolfShrink S T Sigma) := by
  unfold oracleLedoitWolfShrink
  exact shrinkMatrix_symm _ _ _ hS hT

theorem oracleLedoitWolfShrink_psd {k : Nat}
    (S T Sigma : CovMatrix k)
    (hS : PositiveSemidefinite S)
    (hT : PositiveSemidefinite T) :
    PositiveSemidefinite (oracleLedoitWolfShrink S T Sigma) := by
  unfold oracleLedoitWolfShrink
  apply shrinkMatrix_psd _ _ _
  · exact clippedRawShrinkageCoeff_nonneg _ _ _
  · exact clippedRawShrinkageCoeff_le_one _ _ _
  · exact hS
  · exact hT

end FrobeniusLoss

end Covstream
