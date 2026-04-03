import Covstream.Contract

/-!
`Covstream.Examples` is a small collection of worked 2D examples.

These examples show the checked
API boundary and one concrete shrinkage output without requiring a full tour of
the proof development first.
-/

namespace Covstream

namespace Examples

/-- The sample `(1, 2)` written in the runtime array format. -/
def sample12 : Array Real := #[1, 2]

/-- A second sample `(3, 4)` written in the runtime array format. -/
def sample34 : Array Real := #[3, 4]

/-- The exact mean vector after observing `(1, 2)` once. -/
def mean12 : Obs 2 := fun i => if i.1 = 0 then 1 else 2

/-- The exact mean vector after observing `(1, 2)` and `(3, 4)`. -/
def mean23 : Obs 2 := fun i => if i.1 = 0 then 2 else 3

/-- The zero covariance numerator used after a single observation. -/
def zeroCov2 : CovMatrix 2 := fun _ _ => 0

/-- The covariance numerator after the two-point history `(1,2), (3,4)`. -/
def covarianceNumerator22 : CovMatrix 2 := fun _ _ => 2

/-- The Ledoit-Wolf output for the two-point history with shrinkage `α = 1/2`. -/
def shrunkCovarianceHalf : CovMatrix 2 := fun i j => if i = j then 2 else 1

/-- A checked empty 2D state in row-major layout. -/
def twoDimInitState : StreamingContract :=
  { dimension := 2
    positive_dimension := by decide
    layout := .rowMajor
    estimator := WelfordCov.empty 2 }

/--
This is the exact state after observing `(1, 2)` once.

It is the smallest state for which covariance extraction is still unavailable.
-/
def oneSampleState : StreamingContract :=
  { dimension := 2
    positive_dimension := by decide
    layout := .rowMajor
    estimator :=
      { n := 1
        mean := mean12
        C := zeroCov2 } }

/--
This is the exact state after observing `(1, 2)` and `(3, 4)`.

The covariance matrix is the all-`2` matrix, and half-shrinkage toward `μ I`
produces the row-major buffer `#[2, 1, 1, 2]`.
-/
def twoSampleState : StreamingContract :=
  { dimension := 2
    positive_dimension := by decide
    layout := .rowMajor
    estimator :=
      { n := 2
        mean := mean23
        C := covarianceNumerator22 } }

theorem twoDim_init_succeeds :
    StreamingContract.init? 2 .rowMajor = .ok twoDimInitState := by
  simp [twoDimInitState, StreamingContract.init?]

theorem twoDim_wrongDimension_rejected :
    StreamingContract.observe? twoDimInitState #[1]
      = .error (.wrongDimension 2 1) := by
  exact StreamingContract.observe?_eq_wrongDimension twoDimInitState #[1] (by decide)

theorem oneSample_covariance_unavailable :
    StreamingContract.covariance? oneSampleState = .error (.insufficientSamples 1) := by
  exact StreamingContract.covariance?_eq_insufficientSamples oneSampleState (by
    simp [oneSampleState, StreamingContract.sampleCount])

theorem twoSample_covariance_buffer_available :
    ∃ buf, StreamingContract.covarianceBuffer? twoSampleState = .ok buf := by
  refine ⟨StreamingContract.encodeMatrix .rowMajor (covarianceMatrix twoSampleState.estimator), ?_⟩
  rw [StreamingContract.covarianceBuffer?_eq_exact (h := by
    simp [twoSampleState, StreamingContract.sampleCount])]
  rfl

/-- The named shrunk matrix encodes to the expected row-major buffer. -/
theorem shrunkCovarianceHalf_buffer_literal :
    StreamingContract.encodeMatrix .rowMajor shrunkCovarianceHalf = #[2, 1, 1, 2] := by
  rfl

/--
Concrete row-major Ledoit-Wolf extraction.

With `α = 1/2`, the all-`2` covariance matrix shrinks to
`[[2, 1], [1, 2]]`, which is encoded as `#[2, 1, 1, 2]`.
-/
theorem twoSample_ledoitWolf_buffer_available :
    ∃ buf, StreamingContract.ledoitWolfBuffer? twoSampleState (.fixedAlpha (1 / 2)) = .ok buf := by
  refine ⟨StreamingContract.encodeMatrix .rowMajor
      (ledoitWolfShrink (1 / 2) (covarianceMatrix twoSampleState.estimator)), ?_⟩
  rw [StreamingContract.ledoitWolfBuffer?_fixedAlpha_eq (h := by
    simp [twoSampleState, StreamingContract.sampleCount])]
  · rfl

end Examples

end Covstream
