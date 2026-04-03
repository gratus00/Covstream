import Covstream.Welford

/-!
`Covstream.LedoitWolf` builds the structural shrinkage layer on top of the
Welford covariance model.

This file defines:

1. The standard scaled-identity shrinkage target.
2. Convex shrinkage between a covariance matrix and its target.
3. Symmetry and PSD-preservation lemmas.
4. API-ready helpers that clamp user-supplied coefficients into `[0, 1]`.
-/

namespace Covstream

section LedoitWolf

/-- Average diagonal entry of a covariance matrix. -/
noncomputable def diagonalMean {k : Nat} (S : CovMatrix k) : Real :=
  (∑ i : Fin k, S i i) / (k : Real)

/-- Scalar multiple of the identity matrix, represented as a function. -/
def scaledIdentity {k : Nat} (μ : Real) : CovMatrix k :=
  fun i j => if i = j then μ else 0

theorem scaledIdentity_symm {k : Nat}
    (μ : Real) :
    Symmetric (scaledIdentity (k := k) μ) := by
  intro i j
  by_cases h : i = j
  · subst h
    simp [scaledIdentity]
  · have h' : j ≠ i := by
      intro hji
      exact h hji.symm
    simp [scaledIdentity, h, h']

theorem quadraticForm_scaledIdentity {k : Nat}
    (μ : Real) (v : Fin k -> Real) :
    quadraticForm (scaledIdentity (k := k) μ) v = μ * ∑ i : Fin k, (v i) ^ 2 := by
  classical
  unfold quadraticForm scaledIdentity
  simp [pow_two]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem scaledIdentity_psd {k : Nat}
    (μ : Real) (hμ : 0 ≤ μ) :
    PositiveSemidefinite (scaledIdentity (k := k) μ) := by
  intro v
  rw [quadraticForm_scaledIdentity]
  apply mul_nonneg hμ
  apply Finset.sum_nonneg
  intro i _
  exact sq_nonneg (v i)

noncomputable def ledoitWolfTarget {k : Nat} (S : CovMatrix k) : CovMatrix k :=
  scaledIdentity (diagonalMean S)

theorem ledoitWolfTarget_symm {k : Nat}
    (S : CovMatrix k) :
    Symmetric (ledoitWolfTarget S) := by
  simpa [ledoitWolfTarget] using scaledIdentity_symm (k := k) (diagonalMean S)

theorem ledoitWolfTarget_psd {k : Nat}
    (S : CovMatrix k)
    (hμ : 0 ≤ diagonalMean S) :
    PositiveSemidefinite (ledoitWolfTarget S) := by
  simpa [ledoitWolfTarget] using scaledIdentity_psd (k := k) (diagonalMean S) hμ

def shrinkMatrix {k : Nat} (α : Real) (S T : CovMatrix k) : CovMatrix k :=
  fun i j => (1 - α) * S i j + α * T i j

theorem shrinkMatrix_symm {k : Nat}
    (α : Real) (S T : CovMatrix k)
    (hS : Symmetric S) (hT : Symmetric T) :
    Symmetric (shrinkMatrix α S T) := by
  intro i j
  simp [shrinkMatrix, hS i j, hT i j]

theorem quadraticForm_shrinkMatrix {k : Nat}
    (α : Real) (S T : CovMatrix k) (v : Fin k -> Real) :
    quadraticForm (shrinkMatrix α S T) v
      = (1 - α) * quadraticForm S v + α * quadraticForm T v := by
  calc
    quadraticForm (shrinkMatrix α S T) v
        = ∑ i : Fin k, ∑ j : Fin k,
            ((1 - α) * (v i * S i j * v j) + α * (v i * T i j * v j)) := by
          unfold quadraticForm shrinkMatrix
          congr with i
          congr with j
          ring
    _ = (1 - α) * quadraticForm S v + α * quadraticForm T v := by
          unfold quadraticForm
          simp [Finset.mul_sum, Finset.sum_add_distrib]

theorem shrinkMatrix_psd {k : Nat}
    (α : Real) (S T : CovMatrix k)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1)
    (hS : PositiveSemidefinite S)
    (hT : PositiveSemidefinite T) :
    PositiveSemidefinite (shrinkMatrix α S T) := by
  intro v
  rw [quadraticForm_shrinkMatrix]
  have hSv : 0 ≤ quadraticForm S v := hS v
  have hTv : 0 ≤ quadraticForm T v := hT v
  have h1mα : 0 ≤ 1 - α := by linarith
  nlinarith

noncomputable def ledoitWolfShrink {k : Nat}
    (α : Real) (S : CovMatrix k) : CovMatrix k :=
  shrinkMatrix α S (ledoitWolfTarget S)

theorem ledoitWolfShrink_symm {k : Nat}
    (α : Real) (S : CovMatrix k)
    (hS : Symmetric S) :
    Symmetric (ledoitWolfShrink α S) := by
  apply shrinkMatrix_symm α S (ledoitWolfTarget S) hS
  exact ledoitWolfTarget_symm S

theorem ledoitWolfShrink_psd {k : Nat}
    (α : Real) (S : CovMatrix k)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1)
    (hS : PositiveSemidefinite S)
    (hμ : 0 ≤ diagonalMean S) :
    PositiveSemidefinite (ledoitWolfShrink α S) := by
  unfold ledoitWolfShrink
  apply shrinkMatrix_psd α S (ledoitWolfTarget S) hα0 hα1 hS
  exact ledoitWolfTarget_psd S hμ

theorem ledoitWolfShrink_zero {k : Nat}
    (S : CovMatrix k) :
    ledoitWolfShrink 0 S = S := by
  funext i j
  simp [ledoitWolfShrink, shrinkMatrix]

theorem ledoitWolfShrink_one {k : Nat}
    (S : CovMatrix k) :
    ledoitWolfShrink 1 S = ledoitWolfTarget S := by
  funext i j
  simp [ledoitWolfShrink, shrinkMatrix]

/-- Apply shrinkage to the covariance produced by a history of samples. -/
noncomputable def fromListLedoitWolf {k : Nat}
    (α : Real) (xs : List (Fin k -> Real)) : CovMatrix k :=
  ledoitWolfShrink α (covarianceMatrix (fromList k xs))

theorem fromListLedoitWolf_symm {k : Nat}
    (α : Real) (xs : List (Fin k -> Real)) :
    Symmetric (fromListLedoitWolf α xs) := by
  unfold fromListLedoitWolf
  apply ledoitWolfShrink_symm
  exact fromList_covarianceMatrix_symm xs

/-- Clamp any real coefficient into the valid shrinkage interval `[0, 1]`. -/
def clip01 (α : Real) : Real :=
  max 0 (min 1 α)

theorem clip01_nonneg (α : Real) : 0 ≤ clip01 α := by
  unfold clip01
  exact le_max_left _ _

theorem clip01_le_one (α : Real) : clip01 α ≤ 1 := by
  unfold clip01
  by_cases h : 0 ≤ min 1 α
  · simp [max_eq_right h]
  · have h0 : min 1 α < 0 := lt_of_not_ge h
    have hmax : max 0 (min 1 α) = 0 := max_eq_left_of_lt h0
    rw [hmax]
    linarith

theorem clip01_mem_Icc (α : Real) : clip01 α ∈ Set.Icc 0 1 := by
  exact ⟨clip01_nonneg α, clip01_le_one α⟩

/-- API-safe shrinkage: clamp the user-provided coefficient before shrinking. -/
noncomputable def boundedLedoitWolfShrink {k : Nat}
    (α : Real) (S : CovMatrix k) : CovMatrix k :=
  ledoitWolfShrink (clip01 α) S

theorem boundedLedoitWolfShrink_symm {k : Nat}
    (α : Real) (S : CovMatrix k)
    (hS : Symmetric S) :
    Symmetric (boundedLedoitWolfShrink α S) := by
  unfold boundedLedoitWolfShrink
  exact ledoitWolfShrink_symm (clip01 α) S hS

theorem boundedLedoitWolfShrink_psd {k : Nat}
    (α : Real) (S : CovMatrix k)
    (hS : PositiveSemidefinite S)
    (hμ : 0 ≤ diagonalMean S) :
    PositiveSemidefinite (boundedLedoitWolfShrink α S) := by
  unfold boundedLedoitWolfShrink
  apply ledoitWolfShrink_psd (clip01 α) S (clip01_nonneg α) (clip01_le_one α) hS hμ

/-- API-facing shrinkage estimate from a history of samples. -/
noncomputable def fitLedoitWolf {k : Nat}
    (α : Real) (xs : List (Fin k -> Real)) : CovMatrix k :=
  boundedLedoitWolfShrink α (fitCovariance k xs)

theorem fitLedoitWolf_symm {k : Nat}
    (α : Real) (xs : List (Fin k -> Real)) :
    Symmetric (fitLedoitWolf α xs) := by
  unfold fitLedoitWolf
  exact boundedLedoitWolfShrink_symm α _ (fitCovariance_symm xs)

end LedoitWolf

end Covstream
