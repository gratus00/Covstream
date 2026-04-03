import Covstream.Welford

/-!
`Covstream.LedoitWolf` builds the structural shrinkage layer on top of the
exact Welford covariance model.

Read this file in three stages:

1. `ledoitWolfTarget`
2. `ledoitWolfShrink`
3. `fitLedoitWolf_eq_classicalLedoitWolf`

The key idea is simple: shrink a covariance matrix toward the scaled identity
matrix `μ I`, then prove symmetry and PSD are preserved under the expected
assumptions.
-/

namespace Covstream

section LedoitWolf

/-- Average diagonal entry of a covariance matrix. -/
noncomputable def diagonalMean {k : Nat} (S : CovMatrix k) : Real :=
  (∑ i : Fin k, S i i) / (k : Real)

theorem diagonalMean_nonneg_of_psd {k : Nat}
    (hk : 0 < k)
    (S : CovMatrix k)
    (hS : PositiveSemidefinite S) :
    0 ≤ diagonalMean S := by
  unfold diagonalMean
  have hsum : 0 ≤ ∑ i : Fin k, S i i := by
    apply Finset.sum_nonneg
    intro i _
    exact psd_diagonal_nonneg S hS i
  have hkR : 0 < (k : Real) := by
    exact_mod_cast hk
  exact div_nonneg hsum (le_of_lt hkR)

/-- Scalar multiple of the identity matrix, represented as a coordinate function. -/
def scaledIdentity {k : Nat} (μ : Real) : CovMatrix k :=
  fun i j => if i = j then μ else 0

theorem scaledIdentity_isSymmetric {k : Nat}
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
    (μ : Real) (v : Obs k) :
    quadraticForm (scaledIdentity (k := k) μ) v = μ * ∑ i : Fin k, (v i) ^ 2 := by
  classical
  unfold quadraticForm scaledIdentity
  simp [pow_two]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem scaledIdentity_isPsd {k : Nat}
    (μ : Real) (hμ : 0 ≤ μ) :
    PositiveSemidefinite (scaledIdentity (k := k) μ) := by
  intro v
  rw [quadraticForm_scaledIdentity]
  apply mul_nonneg hμ
  apply Finset.sum_nonneg
  intro i _
  exact sq_nonneg (v i)

/-- The standard Ledoit-Wolf target for linear shrinkage: `μ I`. -/
noncomputable def ledoitWolfTarget {k : Nat} (S : CovMatrix k) : CovMatrix k :=
  scaledIdentity (diagonalMean S)

theorem ledoitWolfTarget_isSymmetric {k : Nat}
    (S : CovMatrix k) :
    Symmetric (ledoitWolfTarget S) := by
  simpa [ledoitWolfTarget] using scaledIdentity_isSymmetric (k := k) (diagonalMean S)

theorem ledoitWolfTarget_isPsd {k : Nat}
    (S : CovMatrix k)
    (hμ : 0 ≤ diagonalMean S) :
    PositiveSemidefinite (ledoitWolfTarget S) := by
  simpa [ledoitWolfTarget] using scaledIdentity_isPsd (k := k) (diagonalMean S) hμ

theorem ledoitWolfTarget_isPsd_of_psd {k : Nat}
    (hk : 0 < k)
    (S : CovMatrix k)
    (hS : PositiveSemidefinite S) :
    PositiveSemidefinite (ledoitWolfTarget S) := by
  exact ledoitWolfTarget_isPsd S (diagonalMean_nonneg_of_psd hk S hS)

/-- Convex interpolation between a covariance matrix and a target matrix. -/
def shrinkMatrix {k : Nat} (α : Real) (S T : CovMatrix k) : CovMatrix k :=
  fun i j => (1 - α) * S i j + α * T i j

theorem shrinkMatrix_isSymmetric {k : Nat}
    (α : Real) (S T : CovMatrix k)
    (hS : Symmetric S) (hT : Symmetric T) :
    Symmetric (shrinkMatrix α S T) := by
  intro i j
  simp [shrinkMatrix, hS i j, hT i j]

theorem quadraticForm_shrinkMatrix {k : Nat}
    (α : Real) (S T : CovMatrix k) (v : Obs k) :
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

theorem shrinkMatrix_preservesPsd {k : Nat}
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

/-- Ledoit-Wolf shrinkage toward the scaled-identity target. -/
noncomputable def ledoitWolfShrink {k : Nat}
    (α : Real) (S : CovMatrix k) : CovMatrix k :=
  shrinkMatrix α S (ledoitWolfTarget S)

theorem ledoitWolfShrink_isSymmetric {k : Nat}
    (α : Real) (S : CovMatrix k)
    (hS : Symmetric S) :
    Symmetric (ledoitWolfShrink α S) := by
  apply shrinkMatrix_isSymmetric α S (ledoitWolfTarget S) hS
  exact ledoitWolfTarget_isSymmetric S

theorem ledoitWolfShrink_isPsd {k : Nat}
    (α : Real) (S : CovMatrix k)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1)
    (hS : PositiveSemidefinite S)
    (hμ : 0 ≤ diagonalMean S) :
    PositiveSemidefinite (ledoitWolfShrink α S) := by
  unfold ledoitWolfShrink
  apply shrinkMatrix_preservesPsd α S (ledoitWolfTarget S) hα0 hα1 hS
  exact ledoitWolfTarget_isPsd S hμ

theorem ledoitWolfShrink_isPsd_of_psd {k : Nat}
    (α : Real) (S : CovMatrix k)
    (hk : 0 < k)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1)
    (hS : PositiveSemidefinite S) :
    PositiveSemidefinite (ledoitWolfShrink α S) := by
  exact ledoitWolfShrink_isPsd α S hα0 hα1 hS (diagonalMean_nonneg_of_psd hk S hS)

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

theorem boundedLedoitWolfShrink_isSymmetric {k : Nat}
    (α : Real) (S : CovMatrix k)
    (hS : Symmetric S) :
    Symmetric (boundedLedoitWolfShrink α S) := by
  unfold boundedLedoitWolfShrink
  exact ledoitWolfShrink_isSymmetric (clip01 α) S hS

theorem boundedLedoitWolfShrink_isPsd {k : Nat}
    (α : Real) (S : CovMatrix k)
    (hS : PositiveSemidefinite S)
    (hμ : 0 ≤ diagonalMean S) :
    PositiveSemidefinite (boundedLedoitWolfShrink α S) := by
  unfold boundedLedoitWolfShrink
  apply ledoitWolfShrink_isPsd (clip01 α) S (clip01_nonneg α) (clip01_le_one α) hS hμ

theorem boundedLedoitWolfShrink_isPsd_of_psd {k : Nat}
    (α : Real) (S : CovMatrix k)
    (hk : 0 < k)
    (hS : PositiveSemidefinite S) :
    PositiveSemidefinite (boundedLedoitWolfShrink α S) := by
  exact boundedLedoitWolfShrink_isPsd α S hS (diagonalMean_nonneg_of_psd hk S hS)

/-!
History-facing wrappers.

These are the entry points most relevant to the Rust implementation:
fit a covariance matrix from raw data, then shrink it with a coefficient that
is clamped into the mathematically valid interval.
-/

/-- Apply exact Ledoit-Wolf shrinkage to the covariance produced by a history. -/
noncomputable def fromListLedoitWolf {k : Nat}
    (α : Real) (xs : History k) : CovMatrix k :=
  ledoitWolfShrink α (covarianceMatrix (fromList k xs))

theorem fromListLedoitWolf_isSymmetric {k : Nat}
    (α : Real) (xs : History k) :
    Symmetric (fromListLedoitWolf α xs) := by
  unfold fromListLedoitWolf
  apply ledoitWolfShrink_isSymmetric
  exact fromList_covarianceMatrix_isSymmetric xs

/-- Classical Ledoit-Wolf shrinkage written against the direct classical covariance. -/
noncomputable def classicalLedoitWolf {k : Nat}
    (α : Real) (xs : History k) : CovMatrix k :=
  boundedLedoitWolfShrink α (classicalCovariance xs)

/-- API-facing Ledoit-Wolf estimator built from the streaming covariance fit. -/
noncomputable def fitLedoitWolf {k : Nat}
    (α : Real) (xs : History k) : CovMatrix k :=
  boundedLedoitWolfShrink α (fitCovariance k xs)

theorem fitLedoitWolf_isSymmetric {k : Nat}
    (α : Real) (xs : History k) :
    Symmetric (fitLedoitWolf α xs) := by
  unfold fitLedoitWolf
  exact boundedLedoitWolfShrink_isSymmetric α _ (fitCovariance_isSymmetric xs)

theorem classicalLedoitWolf_isSymmetric {k : Nat}
    (α : Real) (xs : History k) :
    Symmetric (classicalLedoitWolf α xs) := by
  unfold classicalLedoitWolf
  exact boundedLedoitWolfShrink_isSymmetric α _ (classicalCovariance_isSymmetric xs)

/--
Main Ledoit-Wolf correctness theorem at the history level.

The streaming covariance pipeline and the direct classical covariance pipeline
produce the same shrunk estimator in the exact `Real` model.
-/
theorem fitLedoitWolf_eq_classicalLedoitWolf {k : Nat}
    (α : Real) (xs : History k) :
    fitLedoitWolf α xs = classicalLedoitWolf α xs := by
  funext i j
  simp [fitLedoitWolf, classicalLedoitWolf, fitCovariance_eq_classicalCovariance]

theorem classicalLedoitWolf_isPsd_of_psd {k : Nat}
    (α : Real) (xs : History k)
    (hk : 0 < k)
    (hS : PositiveSemidefinite (classicalCovariance xs)) :
    PositiveSemidefinite (classicalLedoitWolf α xs) := by
  unfold classicalLedoitWolf
  exact boundedLedoitWolfShrink_isPsd_of_psd α _ hk hS

theorem fitLedoitWolf_isPsd_of_psd {k : Nat}
    (α : Real) (xs : History k)
    (hk : 0 < k)
    (hS : PositiveSemidefinite (fitCovariance k xs)) :
    PositiveSemidefinite (fitLedoitWolf α xs) := by
  unfold fitLedoitWolf
  exact boundedLedoitWolfShrink_isPsd_of_psd α _ hk hS

end LedoitWolf

end Covstream
