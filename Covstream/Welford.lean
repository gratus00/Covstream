import Mathlib

/-!
`Covstream.Welford` contains the Real-valued online covariance specification.

It defines:

1. The running Welford state.
2. One-step and history-based updates.
3. Symmetry/count invariants.
4. Covariance-matrix vocabulary used by the Ledoit-Wolf layer.
-/

namespace Covstream

/-- Running state for a `k`-dimensional online covariance estimator. -/
structure WelfordCov (k : Nat) where
  n : Nat
  mean : Fin k -> Real
  C : Fin k -> Fin k -> Real

section Welford

def WelfordCov.empty (k : Nat) : WelfordCov k :=
  {
    n := 0
    mean := fun _ => 0
    C := fun _ _ => 0
  }

noncomputable def WelfordCov.observe (est : WelfordCov k) (x : Fin k -> Real) : WelfordCov k :=
  let n' := est.n + 1
  let nf : Real := n'
  let delta := fun i => x i - est.mean i
  let mean' := fun i => est.mean i + delta i / nf
  let C' := fun i j => est.C i j + delta i * (x j - mean' j)
  { n := n', mean := mean', C := C' }

theorem observe_increments_n {k : Nat} (est : WelfordCov k) (x : Fin k → Real) :
    (WelfordCov.observe est x).n = est.n + 1 := by
  rfl

theorem empty_has_zero_observations (k : Nat) :
    (WelfordCov.empty k).n = 0 := by
  rfl

theorem observe_twice_n {k : Nat} (est : WelfordCov k) (x y : Fin k → Real) :
    (WelfordCov.observe (WelfordCov.observe est x) y).n = est.n + 2 := by
  simp [WelfordCov.observe]

theorem empty_C_symm (k : Nat) (i j : Fin k) :
    (WelfordCov.empty k).C i j = (WelfordCov.empty k).C j i := by
  rfl

theorem observe_C_symm {k : Nat} (est : WelfordCov k)
    (x : Fin k -> Real)
    (h : ∀ i j, est.C i j = est.C j i)
    (i j : Fin k) :
    (WelfordCov.observe est x).C i j = (WelfordCov.observe est x).C j i := by
  let n' := est.n + 1
  let nf : Real := n'
  have hnf : nf ≠ 0 := by
    dsimp [nf, n']
    exact_mod_cast Nat.succ_ne_zero est.n
  simp [WelfordCov.observe]
  rw [h j i]
  field_simp [hnf]
  ring

theorem empty_covariance_zero (k : Nat) (i j : Fin k) :
    (WelfordCov.empty k).C i j = 0 := by
  rfl

end Welford

section Histories

/-- Replay a whole history of observations through the online update. -/
noncomputable def observeMany {k : Nat} : WelfordCov k -> List (Fin k -> Real) -> WelfordCov k
| est, [] => est
| est, x :: xs => observeMany (WelfordCov.observe est x) xs

theorem observeMany_increments_n {k : Nat}
    (est : WelfordCov k) (xs : List (Fin k -> Real)) :
    (observeMany est xs).n = est.n + xs.length := by
  induction xs generalizing est with
  | nil =>
      simp [observeMany]
  | cons x xs ih =>
      rw [observeMany]
      rw [ih (WelfordCov.observe est x)]
      rw [observe_increments_n]
      simp [Nat.add_assoc, Nat.add_comm]

noncomputable def fromList (k : Nat) (xs : List (Fin k -> Real)) : WelfordCov k :=
  observeMany (WelfordCov.empty k) xs

theorem observeMany_C_symm {k : Nat}
    (est : WelfordCov k)
    (xs : List (Fin k -> Real))
    (h : ∀ i j, est.C i j = est.C j i)
    (i j : Fin k) :
    (observeMany est xs).C i j = (observeMany est xs).C j i := by
  induction xs generalizing est with
  | nil =>
      simpa [observeMany] using h i j
  | cons x xs ih =>
      rw [observeMany]
      apply ih
      intro a b
      exact observe_C_symm est x h a b

theorem fromList_increments_n {k : Nat}
    (xs : List (Fin k -> Real)) :
    (fromList k xs).n = xs.length := by
  unfold fromList
  simpa [empty_has_zero_observations] using observeMany_increments_n (WelfordCov.empty k) xs

theorem fromList_C_symm {k : Nat}
    (xs : List (Fin k -> Real))
    (i j : Fin k) :
    (fromList k xs).C i j = (fromList k xs).C j i := by
  unfold fromList
  exact observeMany_C_symm (WelfordCov.empty k) xs (empty_C_symm k) i j

end Histories

section CovarianceMatrices

abbrev CovMatrix (k : Nat) := Fin k -> Fin k -> Real

def Symmetric {k : Nat} (S : CovMatrix k) : Prop :=
  ∀ i j, S i j = S j i

def quadraticForm {k : Nat} (S : CovMatrix k) (v : Fin k -> Real) : Real :=
  ∑ i : Fin k, ∑ j : Fin k, v i * S i j * v j

def PositiveSemidefinite {k : Nat} (S : CovMatrix k) : Prop :=
  ∀ v : Fin k -> Real, 0 ≤ quadraticForm S v

def basisVec {k : Nat} (i : Fin k) : Fin k -> Real :=
  fun j => if j = i then 1 else 0

theorem quadraticForm_basisVec {k : Nat}
    (S : CovMatrix k) (i : Fin k) :
    quadraticForm S (basisVec i) = S i i := by
  classical
  simp [quadraticForm, basisVec]

theorem psd_diag_nonneg {k : Nat}
    (S : CovMatrix k)
    (hS : PositiveSemidefinite S)
    (i : Fin k) :
    0 ≤ S i i := by
  simpa [quadraticForm_basisVec] using hS (basisVec i)

/-- Normalize Welford's accumulator by `n - 1` to obtain sample covariance. -/
noncomputable def covarianceMatrix {k : Nat} (est : WelfordCov k) : CovMatrix k :=
  fun i j => est.C i j / ((est.n - 1 : Nat) : Real)

theorem covarianceMatrix_symm {k : Nat}
    (est : WelfordCov k)
    (h : ∀ i j, est.C i j = est.C j i) :
    Symmetric (covarianceMatrix est) := by
  intro i j
  rw [covarianceMatrix, covarianceMatrix]
  rw [h i j]

theorem fromList_covarianceMatrix_symm {k : Nat}
    (xs : List (Fin k -> Real)) :
    Symmetric (covarianceMatrix (fromList k xs)) := by
  intro i j
  exact covarianceMatrix_symm (fromList k xs) (fromList_C_symm xs) i j

/-- Coordinatewise sum of a raw observation history. -/
def rawSum {k : Nat} (xs : List (Fin k -> Real)) : Fin k -> Real :=
  fun i => (xs.map (fun x => x i)).sum

/-- Coordinatewise sum of outer-product entries over a raw observation history. -/
def rawCrossSum {k : Nat} (xs : List (Fin k -> Real)) : CovMatrix k :=
  fun i j => (xs.map (fun x => x i * x j)).sum

/-- Classical sample mean computed directly from the raw history. -/
noncomputable def classicalMean {k : Nat} (xs : List (Fin k -> Real)) : Fin k -> Real :=
  fun i =>
    if xs = [] then 0
    else rawSum xs i / (xs.length : Real)

/--
Classical covariance numerator from the raw history.

This is the usual centered sum written in the equivalent
`Σ xxᵀ - n μ μᵀ` form, which is the right shape for proving
correspondence with Welford's online accumulator.
-/
noncomputable def classicalCovarianceNumerator {k : Nat}
    (xs : List (Fin k -> Real)) : CovMatrix k :=
  fun i j => rawCrossSum xs i j - (xs.length : Real) * classicalMean xs i * classicalMean xs j

/-- Classical unbiased sample covariance from a raw history. -/
noncomputable def classicalCovariance {k : Nat}
    (xs : List (Fin k -> Real)) : CovMatrix k :=
  fun i j => classicalCovarianceNumerator xs i j / ((xs.length - 1 : Nat) : Real)

theorem observeMany_append {k : Nat}
    (est : WelfordCov k)
    (xs ys : List (Fin k -> Real)) :
    observeMany est (xs ++ ys) = observeMany (observeMany est xs) ys := by
  induction xs generalizing est with
  | nil =>
      simp [observeMany]
  | cons x xs ih =>
      simp [observeMany, ih]

theorem fromList_append_singleton {k : Nat}
    (xs : List (Fin k -> Real))
    (x : Fin k -> Real) :
    fromList k (xs ++ [x]) = WelfordCov.observe (fromList k xs) x := by
  unfold fromList
  rw [observeMany_append]
  simp [observeMany]

theorem rawSum_append_singleton {k : Nat}
    (xs : List (Fin k -> Real))
    (x : Fin k -> Real)
    (i : Fin k) :
    rawSum (xs ++ [x]) i = rawSum xs i + x i := by
  simp [rawSum, List.map_append, add_comm]

theorem rawCrossSum_append_singleton {k : Nat}
    (xs : List (Fin k -> Real))
    (x : Fin k -> Real)
    (i j : Fin k) :
    rawCrossSum (xs ++ [x]) i j = rawCrossSum xs i j + x i * x j := by
  simp [rawCrossSum, List.map_append, add_comm]

theorem classicalMean_append_singleton {k : Nat}
    (xs : List (Fin k -> Real))
    (x : Fin k -> Real)
    (i : Fin k) :
    classicalMean (xs ++ [x]) i
      = classicalMean xs i + (x i - classicalMean xs i) / ((xs.length + 1 : Nat) : Real) := by
  by_cases hxs : xs = []
  · subst hxs
    simp [classicalMean, rawSum]
  · have hlenNat : xs.length ≠ 0 := by
      exact Nat.ne_zero_of_lt (List.length_pos_iff_ne_nil.mpr hxs)
    have hlen : ((xs.length : Nat) : Real) ≠ 0 := by
      exact_mod_cast hlenNat
    have hlen1 : (((xs.length + 1 : Nat) : Nat) : Real) ≠ 0 := by
      exact_mod_cast Nat.succ_ne_zero xs.length
    simp [classicalMean, rawSum, hxs, List.map_append]
    field_simp [hlen, hlen1]
    ring

theorem fromList_mean_eq_classicalMean {k : Nat}
    (xs : List (Fin k -> Real))
    (i : Fin k) :
    (fromList k xs).mean i = classicalMean xs i := by
  induction xs using List.reverseRecOn with
  | nil =>
      simp [fromList, observeMany, classicalMean, WelfordCov.empty]
  | append_singleton xs x ih =>
      rw [fromList_append_singleton]
      simp [WelfordCov.observe, fromList_increments_n, classicalMean_append_singleton, ih]

theorem classicalCovarianceNumerator_append_singleton {k : Nat}
    (xs : List (Fin k -> Real))
    (x : Fin k -> Real)
    (i j : Fin k) :
    classicalCovarianceNumerator (xs ++ [x]) i j
      = classicalCovarianceNumerator xs i j
          + (x i - classicalMean xs i) * (x j - classicalMean (xs ++ [x]) j) := by
  have hn1 : (((xs.length + 1 : Nat) : Nat) : Real) ≠ 0 := by
    exact_mod_cast Nat.succ_ne_zero xs.length
  rw [classicalCovarianceNumerator, classicalCovarianceNumerator,
    rawCrossSum_append_singleton, classicalMean_append_singleton xs x i,
    classicalMean_append_singleton xs x j]
  simp
  field_simp [hn1]
  ring

theorem fromList_C_eq_classicalCovarianceNumerator {k : Nat}
    (xs : List (Fin k -> Real))
    (i j : Fin k) :
    (fromList k xs).C i j = classicalCovarianceNumerator xs i j := by
  induction xs using List.reverseRecOn with
  | nil =>
      simp [fromList, observeMany, classicalCovarianceNumerator, rawCrossSum, classicalMean,
        WelfordCov.empty]
  | append_singleton xs x ih =>
      rw [fromList_append_singleton, classicalCovarianceNumerator_append_singleton]
      simp [WelfordCov.observe, fromList_increments_n, ih, fromList_mean_eq_classicalMean,
        classicalMean_append_singleton]

theorem classicalCovarianceNumerator_symm {k : Nat}
    (xs : List (Fin k -> Real)) :
    Symmetric (classicalCovarianceNumerator xs) := by
  intro i j
  simp [classicalCovarianceNumerator, rawCrossSum, mul_comm, mul_left_comm]

theorem classicalCovariance_symm {k : Nat}
    (xs : List (Fin k -> Real)) :
    Symmetric (classicalCovariance xs) := by
  intro i j
  unfold classicalCovariance
  rw [classicalCovarianceNumerator_symm xs i j]

/-- API-facing name for the fitted Welford state from a history. -/
noncomputable def fitWelford (k : Nat) (xs : List (Fin k -> Real)) : WelfordCov k :=
  fromList k xs

/-- API-facing name for the fitted covariance matrix from a history. -/
noncomputable def fitCovariance (k : Nat) (xs : List (Fin k -> Real)) : CovMatrix k :=
  covarianceMatrix (fromList k xs)

theorem fitCovariance_symm {k : Nat} (xs : List (Fin k -> Real)) :
    Symmetric (fitCovariance k xs) := by
  exact fromList_covarianceMatrix_symm xs

theorem fitCovariance_eq_classicalCovariance {k : Nat}
    (xs : List (Fin k -> Real)) :
    fitCovariance k xs = classicalCovariance xs := by
  funext i j
  simp [fitCovariance, covarianceMatrix, classicalCovariance, fromList_C_eq_classicalCovarianceNumerator,
    fromList_increments_n]

end CovarianceMatrices

end Covstream
