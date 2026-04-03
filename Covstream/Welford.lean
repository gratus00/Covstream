import Mathlib

/-!
`Covstream.Welford` is the exact `Real`-valued specification of the streaming
covariance estimator.

Read this file in four stages:

1. `WelfordCov.empty` and `WelfordCov.observe`
2. `observeMany` and `fromList`
3. `covarianceMatrix`
4. `fitCovariance_eq_classicalCovariance`

That last theorem is the headline result of the file: the streaming Welford state
computes the same covariance matrix as the classical batch formula.
-/

namespace Covstream

/-- One exact observation in dimension `k`. -/
abbrev Obs (k : Nat) := Fin k → Real

/-- A finite history of observations. -/
abbrev History (k : Nat) := List (Obs k)

/-- A covariance matrix in coordinate representation. -/
abbrev CovMatrix (k : Nat) := Fin k → Fin k → Real

/-- Running Welford state for a `k`-dimensional covariance estimator. -/
structure WelfordCov (k : Nat) where
  n : Nat
  mean : Obs k
  C : CovMatrix k

section CoreState

/-- The zero-observation Welford state. -/
def WelfordCov.empty (k : Nat) : WelfordCov k :=
  {
    n := 0
    mean := fun _ => 0
    C := fun _ _ => 0
  }

/--
One Welford update step.

`mean` stores the running sample mean. `C` stores the unnormalized covariance
numerator that will later be divided by `n - 1`.
-/
noncomputable def WelfordCov.observe (est : WelfordCov k) (x : Obs k) : WelfordCov k :=
  let n' := est.n + 1
  let nf : Real := n'
  let delta := fun i => x i - est.mean i
  let mean' := fun i => est.mean i + delta i / nf
  let C' := fun i j => est.C i j + delta i * (x j - mean' j)
  { n := n', mean := mean', C := C' }

theorem observe_n_eq_succ {k : Nat} (est : WelfordCov k) (x : Obs k) :
    (WelfordCov.observe est x).n = est.n + 1 := by
  rfl

theorem empty_n_eq_zero (k : Nat) :
    (WelfordCov.empty k).n = 0 := by
  rfl

theorem observe_twice_n_eq_add_two {k : Nat}
    (est : WelfordCov k) (x y : Obs k) :
    (WelfordCov.observe (WelfordCov.observe est x) y).n = est.n + 2 := by
  simp [WelfordCov.observe]

theorem empty_covNumerator_isSymmetric (k : Nat) (i j : Fin k) :
    (WelfordCov.empty k).C i j = (WelfordCov.empty k).C j i := by
  rfl

theorem observe_covNumerator_isSymmetric {k : Nat} (est : WelfordCov k)
    (x : Obs k)
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

theorem empty_covNumerator_eq_zero (k : Nat) (i j : Fin k) :
    (WelfordCov.empty k).C i j = 0 := by
  rfl

end CoreState

section HistoryReplay

/-- Replay a whole history of observations through the streaming update. -/
noncomputable def observeMany {k : Nat} : WelfordCov k → History k → WelfordCov k
| est, [] => est
| est, x :: xs => observeMany (WelfordCov.observe est x) xs

theorem observeMany_n_eq_add_length {k : Nat}
    (est : WelfordCov k) (xs : History k) :
    (observeMany est xs).n = est.n + xs.length := by
  induction xs generalizing est with
  | nil =>
      simp [observeMany]
  | cons x xs ih =>
      rw [observeMany]
      rw [ih (WelfordCov.observe est x)]
      rw [observe_n_eq_succ]
      simp [Nat.add_assoc, Nat.add_comm]

/-- Fit the exact Welford state from a raw history, starting from the empty state. -/
noncomputable def fromList (k : Nat) (xs : History k) : WelfordCov k :=
  observeMany (WelfordCov.empty k) xs

theorem observeMany_covNumerator_isSymmetric {k : Nat}
    (est : WelfordCov k)
    (xs : History k)
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
      exact observe_covNumerator_isSymmetric est x h a b

theorem fromList_n_eq_length {k : Nat}
    (xs : History k) :
    (fromList k xs).n = xs.length := by
  unfold fromList
  simpa [empty_n_eq_zero] using observeMany_n_eq_add_length (WelfordCov.empty k) xs

theorem fromList_covNumerator_isSymmetric {k : Nat}
    (xs : History k)
    (i j : Fin k) :
    (fromList k xs).C i j = (fromList k xs).C j i := by
  unfold fromList
  exact observeMany_covNumerator_isSymmetric (WelfordCov.empty k) xs
    (empty_covNumerator_isSymmetric k) i j

end HistoryReplay

section MatrixVocabulary

/-- Matrix symmetry in the coordinate representation. -/
def Symmetric {k : Nat} (S : CovMatrix k) : Prop :=
  ∀ i j, S i j = S j i

/-- Quadratic form associated to a coordinatewise matrix. -/
def quadraticForm {k : Nat} (S : CovMatrix k) (v : Obs k) : Real :=
  ∑ i : Fin k, ∑ j : Fin k, v i * S i j * v j

/-- Positive semidefiniteness stated through the quadratic form. -/
def PositiveSemidefinite {k : Nat} (S : CovMatrix k) : Prop :=
  ∀ v : Obs k, 0 ≤ quadraticForm S v

/-- Standard basis vector used to read diagonal entries from a quadratic form. -/
def basisVec {k : Nat} (i : Fin k) : Obs k :=
  fun j => if j = i then 1 else 0

theorem quadraticForm_basisVec {k : Nat}
    (S : CovMatrix k) (i : Fin k) :
    quadraticForm S (basisVec i) = S i i := by
  classical
  simp [quadraticForm, basisVec]

theorem psd_diagonal_nonneg {k : Nat}
    (S : CovMatrix k)
    (hS : PositiveSemidefinite S)
    (i : Fin k) :
    0 ≤ S i i := by
  simpa [quadraticForm_basisVec] using hS (basisVec i)

/-- Normalize Welford's accumulator by `n - 1` to obtain sample covariance. -/
noncomputable def covarianceMatrix {k : Nat} (est : WelfordCov k) : CovMatrix k :=
  fun i j => est.C i j / ((est.n - 1 : Nat) : Real)

theorem covarianceMatrix_isSymmetric {k : Nat}
    (est : WelfordCov k)
    (h : ∀ i j, est.C i j = est.C j i) :
    Symmetric (covarianceMatrix est) := by
  intro i j
  rw [covarianceMatrix, covarianceMatrix]
  rw [h i j]

theorem fromList_covarianceMatrix_isSymmetric {k : Nat}
    (xs : History k) :
    Symmetric (covarianceMatrix (fromList k xs)) := by
  intro i j
  exact covarianceMatrix_isSymmetric (fromList k xs) (fromList_covNumerator_isSymmetric xs) i j

end MatrixVocabulary

section ClassicalBatch

/-!
This section is the bridge from the streaming state machine to the direct batch
formulas written from the raw observation history.

The main result is `fitCovariance_eq_classicalCovariance`.
-/

/-- Coordinatewise sum of a raw observation history. -/
def rawSum {k : Nat} (xs : History k) : Obs k :=
  fun i => (xs.map (fun x => x i)).sum

/-- Coordinatewise sum of outer-product entries over a raw observation history. -/
def rawCrossSum {k : Nat} (xs : History k) : CovMatrix k :=
  fun i j => (xs.map (fun x => x i * x j)).sum

/-- Classical sample mean computed directly from the raw history. -/
noncomputable def classicalMean {k : Nat} (xs : History k) : Obs k :=
  fun i =>
    if xs = [] then 0
    else rawSum xs i / (xs.length : Real)

/--
Classical covariance numerator from the raw history.

This is the usual centered sum written in the equivalent
`Σ xxᵀ - n μ μᵀ` form, which matches the shape of Welford's streaming
accumulator.
-/
noncomputable def classicalCovarianceNumerator {k : Nat}
    (xs : History k) : CovMatrix k :=
  fun i j => rawCrossSum xs i j - (xs.length : Real) * classicalMean xs i * classicalMean xs j

/-- Classical unbiased sample covariance from a raw history. -/
noncomputable def classicalCovariance {k : Nat}
    (xs : History k) : CovMatrix k :=
  fun i j => classicalCovarianceNumerator xs i j / ((xs.length - 1 : Nat) : Real)

section Internal

private theorem observeMany_append {k : Nat}
    (est : WelfordCov k)
    (xs ys : History k) :
    observeMany est (xs ++ ys) = observeMany (observeMany est xs) ys := by
  induction xs generalizing est with
  | nil =>
      simp [observeMany]
  | cons x xs ih =>
      simp [observeMany, ih]

private theorem fromList_append_singleton {k : Nat}
    (xs : History k)
    (x : Obs k) :
    fromList k (xs ++ [x]) = WelfordCov.observe (fromList k xs) x := by
  unfold fromList
  rw [observeMany_append]
  simp [observeMany]

private theorem rawSum_append_singleton {k : Nat}
    (xs : History k)
    (x : Obs k)
    (i : Fin k) :
    rawSum (xs ++ [x]) i = rawSum xs i + x i := by
  simp [rawSum, List.map_append, add_comm]

private theorem rawCrossSum_append_singleton {k : Nat}
    (xs : History k)
    (x : Obs k)
    (i j : Fin k) :
    rawCrossSum (xs ++ [x]) i j = rawCrossSum xs i j + x i * x j := by
  simp [rawCrossSum, List.map_append, add_comm]

private theorem classicalMean_append_singleton {k : Nat}
    (xs : History k)
    (x : Obs k)
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

private theorem classicalCovarianceNumerator_append_singleton {k : Nat}
    (xs : History k)
    (x : Obs k)
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

end Internal

/-- Welford's running mean matches the direct batch mean from the same history. -/
theorem fitMean_eq_classicalMean {k : Nat}
    (xs : History k)
    (i : Fin k) :
    (fromList k xs).mean i = classicalMean xs i := by
  induction xs using List.reverseRecOn with
  | nil =>
      simp [fromList, observeMany, classicalMean, WelfordCov.empty]
  | append_singleton xs x ih =>
      rw [fromList_append_singleton]
      simp [WelfordCov.observe, fromList_n_eq_length, classicalMean_append_singleton, ih]

/--
Welford's stored covariance numerator matches the classical batch numerator.

This is the key unnormalized statement that feeds into the final covariance
equality theorem.
-/
theorem fitCovarianceNumerator_eq_classicalNumerator {k : Nat}
    (xs : History k)
    (i j : Fin k) :
    (fromList k xs).C i j = classicalCovarianceNumerator xs i j := by
  induction xs using List.reverseRecOn with
  | nil =>
      simp [fromList, observeMany, classicalCovarianceNumerator, rawCrossSum, classicalMean,
        WelfordCov.empty]
  | append_singleton xs x ih =>
      rw [fromList_append_singleton, classicalCovarianceNumerator_append_singleton]
      simp [WelfordCov.observe, fromList_n_eq_length, ih, fitMean_eq_classicalMean,
        classicalMean_append_singleton]

theorem classicalCovarianceNumerator_isSymmetric {k : Nat}
    (xs : History k) :
    Symmetric (classicalCovarianceNumerator xs) := by
  intro i j
  simp [classicalCovarianceNumerator, rawCrossSum, mul_comm, mul_left_comm]

theorem classicalCovariance_isSymmetric {k : Nat}
    (xs : History k) :
    Symmetric (classicalCovariance xs) := by
  intro i j
  unfold classicalCovariance
  rw [classicalCovarianceNumerator_isSymmetric xs i j]

/-- API-facing name for the fitted Welford state from a history. -/
noncomputable def fitWelford (k : Nat) (xs : History k) : WelfordCov k :=
  fromList k xs

/-- API-facing name for the fitted covariance matrix from a history. -/
noncomputable def fitCovariance (k : Nat) (xs : History k) : CovMatrix k :=
  covarianceMatrix (fromList k xs)

theorem fitCovariance_isSymmetric {k : Nat} (xs : History k) :
    Symmetric (fitCovariance k xs) := by
  exact fromList_covarianceMatrix_isSymmetric xs

/--
Main Welford correctness theorem.

The covariance returned by the streaming state matches the direct classical batch
covariance computed from the same observation history.
-/
theorem fitCovariance_eq_classicalCovariance {k : Nat}
    (xs : History k) :
    fitCovariance k xs = classicalCovariance xs := by
  funext i j
  simp [fitCovariance, covarianceMatrix, classicalCovariance,
    fitCovarianceNumerator_eq_classicalNumerator, fromList_n_eq_length]

end ClassicalBatch

end Covstream
