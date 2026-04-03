import Covstream.LedoitWolf
import Covstream.FrobeniusBasic

/-!
`Covstream.ShrinkageOptimization` studies which shrinkage coefficient is best
for Frobenius loss.

Read this file in three stages:

1. `shrinkageLoss`
2. `rawShrinkageCoeff` and `rawShrinkageCoeff_minimizes_shrinkageLoss`
3. `clippedCoeff_minimizes_shrinkageLoss_on_unitInterval` and
   `oracleLedoitWolfCoeff_minimizes_shrinkageLoss_on_unitInterval`
-/

namespace Covstream

section ShrinkageOptimization

/-- Residual matrix after shrinking `S` toward `T` and comparing with `Sigma`. -/
def shrinkageResidual {k : Nat}
    (α : Real) (S T Sigma : CovMatrix k) : CovMatrix k :=
  matrixSub (shrinkMatrix α S T) Sigma

/-- Direction in which shrinkage moves the estimator. -/
def shrinkageDirection {k : Nat}
    (S T : CovMatrix k) : CovMatrix k :=
  matrixSub T S

/-- Estimation error of `S` relative to the target truth `Sigma`. -/
def estimationError {k : Nat}
    (S Sigma : CovMatrix k) : CovMatrix k :=
  matrixSub S Sigma

/-- Frobenius loss of shrinking `S` toward `T` when the truth is `Sigma`. -/
def shrinkageLoss {k : Nat}
    (α : Real) (S T Sigma : CovMatrix k) : Real :=
  frobeniusNormSq (shrinkageResidual α S T Sigma)

/-- Unconstrained quadratic minimizer of the shrinkage loss. -/
noncomputable def rawShrinkageCoeff {k : Nat}
    (S T Sigma : CovMatrix k) : Real :=
  - frobeniusInner (estimationError S Sigma) (shrinkageDirection S T)
    / frobeniusNormSq (shrinkageDirection S T)

theorem shrinkageResidual_eq_error_plus_direction {k : Nat}
    (α : Real) (S T Sigma : CovMatrix k) :
    shrinkageResidual α S T Sigma
      = matrixAdd (estimationError S Sigma) (matrixSmul α (shrinkageDirection S T)) := by
  funext i j
  simp [shrinkageResidual, estimationError, shrinkageDirection, matrixAdd, matrixSub, matrixSmul,
    shrinkMatrix]
  ring

theorem shrinkageLoss_eq_quadratic {k : Nat}
    (α : Real) (S T Sigma : CovMatrix k) :
    shrinkageLoss α S T Sigma
      = frobeniusNormSq (estimationError S Sigma)
        + 2 * α * frobeniusInner (estimationError S Sigma) (shrinkageDirection S T)
        + α ^ 2 * frobeniusNormSq (shrinkageDirection S T) := by
  unfold shrinkageLoss
  rw [shrinkageResidual_eq_error_plus_direction]
  simpa using
    (frobeniusNormSq_add_smul_eq_quadratic (estimationError S Sigma) (shrinkageDirection S T) α)

theorem rawShrinkageCoeff_minimizes_shrinkageLoss {k : Nat}
    (S T Sigma : CovMatrix k)
    (hdir : frobeniusNormSq (shrinkageDirection S T) ≠ 0)
    (α : Real) :
    shrinkageLoss (rawShrinkageCoeff S T Sigma) S T Sigma ≤ shrinkageLoss α S T Sigma := by
  have hDnonneg : 0 ≤ frobeniusNormSq (shrinkageDirection S T) :=
    frobeniusNormSq_nonneg (shrinkageDirection S T)
  rw [shrinkageLoss_eq_quadratic, shrinkageLoss_eq_quadratic]
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

/-- Completed-square form of the shrinkage loss around the unconstrained optimum. -/
theorem shrinkageLoss_eq_completedSquare {k : Nat}
    (S T Sigma : CovMatrix k)
    (hdir : frobeniusNormSq (shrinkageDirection S T) ≠ 0)
    (α : Real) :
    shrinkageLoss α S T Sigma
      = shrinkageLoss (rawShrinkageCoeff S T Sigma) S T Sigma
          + frobeniusNormSq (shrinkageDirection S T)
              * (α - rawShrinkageCoeff S T Sigma) ^ 2 := by
  let a := frobeniusNormSq (shrinkageDirection S T)
  let b := frobeniusInner (estimationError S Sigma) (shrinkageDirection S T)
  have ha_ne : a ≠ 0 := by
    simpa [a] using hdir
  have hraw : rawShrinkageCoeff S T Sigma = -b / a := by
    simp [rawShrinkageCoeff, a, b]
  rw [shrinkageLoss_eq_quadratic, shrinkageLoss_eq_quadratic, hraw]
  field_simp [ha_ne]
  ring

section Internal

private theorem clip01_sub_sq_le_sub_sq (x α : Real)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) :
    (clip01 x - x) ^ 2 ≤ (α - x) ^ 2 := by
  by_cases hx0 : x < 0
  · have hclip : clip01 x = 0 := by
      have hx1 : x ≤ 1 := le_trans (le_of_lt hx0) zero_le_one
      simp [clip01, min_eq_right hx1, max_eq_left_of_lt hx0]
    rw [hclip]
    nlinarith
  · have hx0' : 0 ≤ x := le_of_not_gt hx0
    by_cases hx1 : x ≤ 1
    · have hclip : clip01 x = x := by
        unfold clip01
        rw [min_eq_right hx1, max_eq_right hx0']
      rw [hclip]
      nlinarith
    · have hx1' : 1 < x := lt_of_not_ge hx1
      have hclip : clip01 x = 1 := by
        unfold clip01
        rw [min_eq_left (le_of_lt hx1'), max_eq_right zero_le_one]
      rw [hclip]
      nlinarith

end Internal

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

/--
Best valid coefficient on the Ledoit-Wolf interval `[0, 1]`.

This upgrades the unconstrained optimum to the coefficient range used in
practice.
-/
theorem clippedCoeff_minimizes_shrinkageLoss_on_unitInterval {k : Nat}
    (S T Sigma : CovMatrix k)
    (hdir : frobeniusNormSq (shrinkageDirection S T) ≠ 0)
    (α : Real)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) :
    shrinkageLoss (clippedRawShrinkageCoeff S T Sigma) S T Sigma ≤ shrinkageLoss α S T Sigma := by
  have hDnonneg : 0 ≤ frobeniusNormSq (shrinkageDirection S T) :=
    frobeniusNormSq_nonneg (shrinkageDirection S T)
  rw [shrinkageLoss_eq_completedSquare S T Sigma hdir (clippedRawShrinkageCoeff S T Sigma),
    shrinkageLoss_eq_completedSquare S T Sigma hdir α]
  have hclip :
      (clippedRawShrinkageCoeff S T Sigma - rawShrinkageCoeff S T Sigma) ^ 2
        ≤ (α - rawShrinkageCoeff S T Sigma) ^ 2 := by
    unfold clippedRawShrinkageCoeff
    exact clip01_sub_sq_le_sub_sq (rawShrinkageCoeff S T Sigma) α hα0 hα1
  nlinarith

/-- Generic clipped oracle shrinkage for an arbitrary target `T`. -/
noncomputable def clippedOracleShrink {k : Nat}
    (S T Sigma : CovMatrix k) : CovMatrix k :=
  shrinkMatrix (clippedRawShrinkageCoeff S T Sigma) S T

theorem clippedOracleShrink_isSymmetric {k : Nat}
    (S T Sigma : CovMatrix k)
    (hS : Symmetric S) (hT : Symmetric T) :
    Symmetric (clippedOracleShrink S T Sigma) := by
  unfold clippedOracleShrink
  exact shrinkMatrix_isSymmetric _ _ _ hS hT

theorem clippedOracleShrink_isPsd {k : Nat}
    (S T Sigma : CovMatrix k)
    (hS : PositiveSemidefinite S)
    (hT : PositiveSemidefinite T) :
    PositiveSemidefinite (clippedOracleShrink S T Sigma) := by
  unfold clippedOracleShrink
  apply shrinkMatrix_preservesPsd _ _ _
  · exact clippedRawShrinkageCoeff_nonneg _ _ _
  · exact clippedRawShrinkageCoeff_le_one _ _ _
  · exact hS
  · exact hT

/-- Oracle Ledoit-Wolf coefficient for the scaled-identity target. -/
noncomputable def oracleLedoitWolfCoeff {k : Nat}
    (S Sigma : CovMatrix k) : Real :=
  clippedRawShrinkageCoeff S (ledoitWolfTarget S) Sigma

theorem oracleLedoitWolfCoeff_nonneg {k : Nat}
    (S Sigma : CovMatrix k) :
    0 ≤ oracleLedoitWolfCoeff S Sigma := by
  unfold oracleLedoitWolfCoeff
  exact clippedRawShrinkageCoeff_nonneg _ _ _

theorem oracleLedoitWolfCoeff_le_one {k : Nat}
    (S Sigma : CovMatrix k) :
    oracleLedoitWolfCoeff S Sigma ≤ 1 := by
  unfold oracleLedoitWolfCoeff
  exact clippedRawShrinkageCoeff_le_one _ _ _

/-- Oracle Ledoit-Wolf estimate specialized to the scaled-identity target. -/
noncomputable def oracleLedoitWolfEstimate {k : Nat}
    (S Sigma : CovMatrix k) : CovMatrix k :=
  ledoitWolfShrink (oracleLedoitWolfCoeff S Sigma) S

theorem oracleLedoitWolfEstimate_isSymmetric {k : Nat}
    (S Sigma : CovMatrix k)
    (hS : Symmetric S) :
    Symmetric (oracleLedoitWolfEstimate S Sigma) := by
  unfold oracleLedoitWolfEstimate oracleLedoitWolfCoeff
  exact ledoitWolfShrink_isSymmetric _ _ hS

theorem oracleLedoitWolfEstimate_isPsd {k : Nat}
    (hk : 0 < k)
    (S Sigma : CovMatrix k)
    (hS : PositiveSemidefinite S) :
    PositiveSemidefinite (oracleLedoitWolfEstimate S Sigma) := by
  unfold oracleLedoitWolfEstimate oracleLedoitWolfCoeff
  exact ledoitWolfShrink_isPsd_of_psd _ _ hk
    (clippedRawShrinkageCoeff_nonneg _ _ _)
    (clippedRawShrinkageCoeff_le_one _ _ _)
    hS

/--
Oracle optimality theorem specialized to the Ledoit-Wolf target.

Among all admissible coefficients `α ∈ [0,1]`, the oracle coefficient minimizes
the Frobenius shrinkage loss.
-/
theorem oracleLedoitWolfCoeff_minimizes_shrinkageLoss_on_unitInterval {k : Nat}
    (S Sigma : CovMatrix k)
    (hdir : frobeniusNormSq (shrinkageDirection S (ledoitWolfTarget S)) ≠ 0)
    (α : Real)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) :
    shrinkageLoss (oracleLedoitWolfCoeff S Sigma) S (ledoitWolfTarget S) Sigma
      ≤ shrinkageLoss α S (ledoitWolfTarget S) Sigma := by
  unfold oracleLedoitWolfCoeff
  exact clippedCoeff_minimizes_shrinkageLoss_on_unitInterval
    S (ledoitWolfTarget S) Sigma hdir α hα0 hα1

end ShrinkageOptimization

end Covstream
