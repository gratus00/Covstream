import Covstream.ShrinkageOptimization

/-!
`Covstream.ErrorBounds` contains abstract perturbation lemmas for the exact
matrix model.

This is not a full floating-point proof. Instead, it provides the kind of
stability statements a future `f64` refinement can plug into:

* if a covariance matrix changes by at most `ε` entrywise, how much can the
  target change?
* how does that perturbation propagate through shrinkage?
* how sensitive is shrinkage to coefficient error?

These are the first useful pieces of an eventual floating-point error analysis.
-/

namespace Covstream

section ErrorBounds

/-- Entrywise `ε`-approximation between two matrices. -/
def EntrywiseApprox {k : Nat} (ε : Real) (A B : CovMatrix k) : Prop :=
  ∀ i j, |A i j - B i j| ≤ ε

/-- Uniform entrywise magnitude bound for a matrix. -/
def EntrywiseBound {k : Nat} (B : Real) (A : CovMatrix k) : Prop :=
  ∀ i j, |A i j| ≤ B

/--
The diagonal average changes by at most the entrywise perturbation level.

This is the scalar fact behind stability of the Ledoit-Wolf target `μ I`.
-/
theorem diagonalMean_errorBound {k : Nat}
    (hk : 0 < k)
    {ε : Real}
    {A B : CovMatrix k}
    (hAB : EntrywiseApprox ε A B) :
    |diagonalMean A - diagonalMean B| ≤ ε := by
  have hkR : 0 < (k : Real) := by
    exact_mod_cast hk
  have hkR0 : (k : Real) ≠ 0 := by
    exact_mod_cast (Nat.ne_of_gt hk)
  have hsum :
      |∑ i : Fin k, (A i i - B i i)| ≤ (k : Real) * ε := by
    calc
      |∑ i : Fin k, (A i i - B i i)| ≤ ∑ i : Fin k, |A i i - B i i| := by
          simpa using (Finset.abs_sum_le_sum_abs (s := Finset.univ) (f := fun i : Fin k => A i i - B i i))
      _ ≤ ∑ _i : Fin k, ε := by
          apply Finset.sum_le_sum
          intro i _
          exact hAB i i
      _ = (k : Real) * ε := by
          simp [mul_comm]
  have hrewrite :
      diagonalMean A - diagonalMean B
        = ((∑ i : Fin k, A i i) - ∑ i : Fin k, B i i) / (k : Real) := by
    unfold diagonalMean
    field_simp [hkR0]
  have hsumRewrite :
      ((∑ i : Fin k, A i i) - ∑ i : Fin k, B i i)
        = ∑ i : Fin k, (A i i - B i i) := by
    rw [← Finset.sum_sub_distrib]
  rw [hrewrite, hsumRewrite, abs_div, abs_of_pos hkR]
  exact (div_le_iff₀ hkR).2 (by simpa [mul_comm] using hsum)

/-- Entrywise perturbations propagate directly to the scaled-identity target. -/
theorem ledoitWolfTarget_entrywiseApprox {k : Nat}
    (hk : 0 < k)
    {ε : Real}
    {A B : CovMatrix k}
    (hAB : EntrywiseApprox ε A B) :
    EntrywiseApprox ε (ledoitWolfTarget A) (ledoitWolfTarget B) := by
  intro i j
  have hε : 0 ≤ ε := le_trans (abs_nonneg (A i i - B i i)) (hAB i i)
  by_cases h : i = j
  · subst h
    simpa [ledoitWolfTarget, scaledIdentity] using diagonalMean_errorBound hk hAB
  · have h' : j ≠ i := by
      intro hji
      exact h hji.symm
    simp [ledoitWolfTarget, scaledIdentity, h, hε]

/--
Shrinkage is Lipschitz in both its covariance input and its target input, with
the expected coefficient weights.
-/
theorem shrinkMatrix_entrywiseApprox {k : Nat}
    (α : Real)
    {εS εT : Real}
    {S S' T T' : CovMatrix k}
    (hS : EntrywiseApprox εS S S')
    (hT : EntrywiseApprox εT T T') :
    EntrywiseApprox (|1 - α| * εS + |α| * εT)
      (shrinkMatrix α S T) (shrinkMatrix α S' T') := by
  intro i j
  have hrewrite :
      shrinkMatrix α S T i j - shrinkMatrix α S' T' i j
        = (1 - α) * (S i j - S' i j) + α * (T i j - T' i j) := by
    simp [shrinkMatrix]
    ring
  rw [hrewrite]
  calc
    |(1 - α) * (S i j - S' i j) + α * (T i j - T' i j)|
        ≤ |(1 - α) * (S i j - S' i j)| + |α * (T i j - T' i j)| := abs_add_le _ _
    _ = |1 - α| * |S i j - S' i j| + |α| * |T i j - T' i j| := by
        rw [abs_mul, abs_mul]
    _ ≤ |1 - α| * εS + |α| * εT := by
        nlinarith [abs_nonneg (1 - α), abs_nonneg α, hS i j, hT i j]

/-- Specialization of `shrinkMatrix_entrywiseApprox` to the Ledoit-Wolf target. -/
theorem ledoitWolfShrink_entrywiseApprox {k : Nat}
    (α : Real)
    (hk : 0 < k)
    {ε : Real}
    {A B : CovMatrix k}
    (hAB : EntrywiseApprox ε A B) :
    EntrywiseApprox ((|1 - α| + |α|) * ε)
      (ledoitWolfShrink α A) (ledoitWolfShrink α B) := by
  have htarget : EntrywiseApprox ε (ledoitWolfTarget A) (ledoitWolfTarget B) :=
    ledoitWolfTarget_entrywiseApprox hk hAB
  have hshrink :
      EntrywiseApprox (|1 - α| * ε + |α| * ε)
        (ledoitWolfShrink α A) (ledoitWolfShrink α B) := by
    simpa [ledoitWolfShrink] using shrinkMatrix_entrywiseApprox α hAB htarget
  intro i j
  have hij := hshrink i j
  convert hij using 1
  ring

/-- On the valid interval, the Ledoit-Wolf shrinkage map is entrywise `ε`-stable. -/
theorem ledoitWolfShrink_entrywiseApprox_of_unitInterval {k : Nat}
    (α : Real)
    (hα0 : 0 ≤ α)
    (hα1 : α ≤ 1)
    (hk : 0 < k)
    {ε : Real}
    {A B : CovMatrix k}
    (hAB : EntrywiseApprox ε A B) :
    EntrywiseApprox ε (ledoitWolfShrink α A) (ledoitWolfShrink α B) := by
  have hbase := ledoitWolfShrink_entrywiseApprox α hk hAB
  have habs : |1 - α| + |α| = 1 := by
    rw [abs_of_nonneg (sub_nonneg.mpr hα1), abs_of_nonneg hα0]
    ring
  simpa [habs] using hbase

/--
Main stability theorem of this file.

The clipped runtime-facing shrinkage map is entrywise `ε`-stable because the
clamped coefficient always lies in the valid interval `[0,1]`.
-/
theorem boundedLedoitWolfShrink_entrywiseApprox {k : Nat}
    (α : Real)
    (hk : 0 < k)
    {ε : Real}
    {A B : CovMatrix k}
    (hAB : EntrywiseApprox ε A B) :
    EntrywiseApprox ε (boundedLedoitWolfShrink α A) (boundedLedoitWolfShrink α B) := by
  unfold boundedLedoitWolfShrink
  exact ledoitWolfShrink_entrywiseApprox_of_unitInterval (clip01 α)
    (clip01_nonneg α) (clip01_le_one α) hk hAB

/--
If the shrinkage direction `T - S` is entrywise bounded, then coefficient error
produces a linear entrywise output error.
-/
theorem shrinkMatrix_coeffError_entrywiseApprox {k : Nat}
    {α β B : Real}
    {S T : CovMatrix k}
    (hBound : EntrywiseBound B (matrixSub T S)) :
    EntrywiseApprox (|α - β| * B)
      (shrinkMatrix α S T) (shrinkMatrix β S T) := by
  intro i j
  have hrewrite :
      shrinkMatrix α S T i j - shrinkMatrix β S T i j
        = (α - β) * (T i j - S i j) := by
    simp [shrinkMatrix]
    ring
  rw [hrewrite, abs_mul]
  exact mul_le_mul_of_nonneg_left (hBound i j) (abs_nonneg _)

end ErrorBounds

end Covstream
