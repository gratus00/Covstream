import Covstream.LedoitWolf

/-!
`Covstream.Contract` is a runtime-facing contract layer for the exact Lean
specification.

The goal of this file is not to introduce floating-point numerics. Instead, it
packages the exact `Real` model into an API shape that a Rust or C++
implementation can mirror directly:

* checked construction
* checked observation updates
* checked covariance extraction
* explicit matrix layout choices

This makes the eventual systems implementation a faithful copy of a precise
specification rather than an independently designed API.
-/

namespace Covstream

section BoundaryTypes

/-- Errors a runtime implementation should expose at the API boundary. -/
inductive CovstreamError where
  /-- The caller attempted to create a zero-dimensional estimator. -/
  | zeroDimension
  /-- The supplied observation length does not match the estimator dimension. -/
  | wrongDimension (expected got : Nat)
  /-- A covariance matrix was requested before enough samples had been observed. -/
  | insufficientSamples (actual : Nat)
  /--
  The runtime implementation received a NaN or infinite input.

  This constructor is included in the contract for Rust/C++ parity. It is not
  produced by the exact `Real`-valued operations in this file.
  -/
  | nonFiniteInput
  deriving Repr, DecidableEq

/-- Concrete buffer layouts supported by the runtime API. -/
inductive MatrixLayout where
  | rowMajor
  | upperTrianglePacked
  deriving Repr, DecidableEq

/-- Number of scalar outputs produced by a given matrix layout. -/
def MatrixLayout.outputSize : MatrixLayout → Nat → Nat
  | .rowMajor, k => k * k
  | .upperTrianglePacked, k => k * (k + 1) / 2

/-- Shrinkage modes exposed at the runtime boundary. -/
inductive ShrinkageMode where
  | fixedAlpha (α : Real)
  | clippedAlpha (α : Real)

end BoundaryTypes

section CheckedState

/-- A checked streaming covariance state with a runtime-facing dimension and layout. -/
structure StreamingContract where
  dimension : Nat
  positive_dimension : 0 < dimension
  layout : MatrixLayout
  estimator : WelfordCov dimension

namespace StreamingContract

/-- Current sample count stored in the streaming state. -/
def sampleCount (state : StreamingContract) : Nat :=
  state.estimator.n

section Internal

private def arrayToObs {k : Nat} (xs : Array Real) (h : xs.size = k) : Obs k :=
  fun i => xs[i.1]'(by simp [h] at i ⊢)

private def obsOfArray? (k : Nat) (xs : Array Real) : Except CovstreamError (Obs k) :=
  if h : xs.size = k then
    .ok (arrayToObs xs h)
  else
    .error (.wrongDimension k xs.size)

private def rowMajorEntriesAux {k : Nat} (rows : List (Fin k)) (S : CovMatrix k) : List Real :=
  match rows with
  | [] => []
  | i :: rows => (List.finRange k).map (fun j => S i j) ++ rowMajorEntriesAux rows S

private def rowMajorEntries {k : Nat} (S : CovMatrix k) : List Real :=
  rowMajorEntriesAux (List.finRange k) S

private def upperTriangleEntries : {k : Nat} → CovMatrix k → List Real
  | 0, _ => []
  | Nat.succ k, S =>
      let i0 : Fin (Nat.succ k) := ⟨0, Nat.succ_pos _⟩
      let firstRow := (List.finRange (Nat.succ k)).map fun j => S i0 j
      let tail : CovMatrix k := fun i j => S i.succ j.succ
      firstRow ++ upperTriangleEntries tail

private theorem rowMajorEntries_length {k : Nat} (S : CovMatrix k) :
    (rowMajorEntries S).length = MatrixLayout.outputSize .rowMajor k := by
  have haux :
      ∀ rows : List (Fin k),
        (rowMajorEntriesAux rows S).length = rows.length * k := by
    intro rows
    induction rows with
    | nil =>
        simp [rowMajorEntriesAux]
    | cons i rows ih =>
        simp [rowMajorEntriesAux, ih, Nat.succ_mul, Nat.add_comm, List.length_finRange]
  simpa [rowMajorEntries, MatrixLayout.outputSize, List.length_finRange] using
    haux (List.finRange k)

private theorem upperTriangleEntries_length :
    ∀ {k : Nat} (S : CovMatrix k),
      (upperTriangleEntries S).length = MatrixLayout.outputSize .upperTrianglePacked k := by
  have hsizeSucc :
      ∀ k : Nat,
        MatrixLayout.outputSize .upperTrianglePacked (k + 1)
          = (k + 1) + MatrixLayout.outputSize .upperTrianglePacked k := by
    intro k
    calc
      MatrixLayout.outputSize .upperTrianglePacked (k + 1)
          = (k + 2).choose 2 := by
              rw [MatrixLayout.outputSize, Nat.choose_two_right]
              simp [Nat.mul_comm, Nat.add_comm, Nat.add_left_comm]
      _ = (k + 1).choose 1 + (k + 1).choose 2 := by
              rw [Nat.choose_succ_succ]
      _ = (k + 1) + MatrixLayout.outputSize .upperTrianglePacked k := by
              rw [Nat.choose_one_right, Nat.choose_two_right]
              simp [MatrixLayout.outputSize, Nat.mul_comm]
  intro k
  induction k with
  | zero =>
      intro S
      simp [upperTriangleEntries, MatrixLayout.outputSize]
  | succ k ih =>
      intro S
      let tail : CovMatrix k := fun i j => S i.succ j.succ
      simpa [hsizeSucc k, upperTriangleEntries, List.length_finRange] using congrArg (fun n => k + 1 + n) (ih tail)

end Internal

section CheckedApi

/-- Create a checked streaming covariance state. -/
def init? (dimension : Nat) (layout : MatrixLayout := .rowMajor) :
    Except CovstreamError StreamingContract :=
  if h : 0 < dimension then
    .ok
      { dimension := dimension
        positive_dimension := h
        layout := layout
        estimator := WelfordCov.empty dimension }
  else
    .error .zeroDimension

/-- Observe one validated `Real`-valued sample. -/
noncomputable def observe? (state : StreamingContract) (xs : Array Real) :
    Except CovstreamError StreamingContract := do
  let obs ← obsOfArray? state.dimension xs
  pure { state with estimator := WelfordCov.observe state.estimator obs }

/-- Extract the exact covariance matrix when enough samples are available. -/
noncomputable def covariance? (state : StreamingContract) :
    Except CovstreamError (CovMatrix state.dimension) :=
  if _h : 2 ≤ state.sampleCount then
    .ok (covarianceMatrix state.estimator)
  else
    .error (.insufficientSamples state.sampleCount)

/-- Apply Ledoit-Wolf shrinkage to the exact covariance matrix. -/
noncomputable def ledoitWolf? (state : StreamingContract) (mode : ShrinkageMode) :
    Except CovstreamError (CovMatrix state.dimension) := do
  let S ← covariance? state
  pure <|
    match mode with
    | .fixedAlpha α => ledoitWolfShrink α S
    | .clippedAlpha α => boundedLedoitWolfShrink α S

/-- Encode a matrix as a dense row-major buffer. -/
def encodeRowMajor {k : Nat} (S : CovMatrix k) : Array Real :=
  (rowMajorEntries S).toArray

/-- Encode the upper triangle of a matrix in packed `(i,j)` order with `i ≤ j`. -/
def encodeUpperTrianglePacked {k : Nat} (S : CovMatrix k) : Array Real :=
  (upperTriangleEntries S).toArray

/-- Encode a matrix using the layout stored in the contract state. -/
def encodeMatrix (layout : MatrixLayout) {k : Nat} (S : CovMatrix k) : Array Real :=
  match layout with
  | .rowMajor => encodeRowMajor S
  | .upperTrianglePacked => encodeUpperTrianglePacked S

/-- Checked covariance extraction in the contract's chosen buffer layout. -/
noncomputable def covarianceBuffer? (state : StreamingContract) :
    Except CovstreamError (Array Real) := do
  let S ← covariance? state
  pure (encodeMatrix state.layout S)

/-- Checked Ledoit-Wolf extraction in the contract's chosen buffer layout. -/
noncomputable def ledoitWolfBuffer? (state : StreamingContract) (mode : ShrinkageMode) :
    Except CovstreamError (Array Real) := do
  let S ← ledoitWolf? state mode
  pure (encodeMatrix state.layout S)

end CheckedApi

section ApiFacts

section ConstructorFacts

theorem init?_eq_zeroDimension (layout : MatrixLayout := .rowMajor) :
    init? 0 layout = .error .zeroDimension := by
  simp [init?]

theorem init?_eq_ok
    (dimension : Nat)
    (layout : MatrixLayout := .rowMajor)
    (h : 0 < dimension) :
    init? dimension layout
      = .ok
          { dimension := dimension
            positive_dimension := h
            layout := layout
            estimator := WelfordCov.empty dimension } := by
  simp [init?, h]

end ConstructorFacts

section UpdateFacts

theorem observe?_eq_wrongDimension
    (state : StreamingContract)
    (xs : Array Real)
    (h : xs.size ≠ state.dimension) :
    observe? state xs = .error (.wrongDimension state.dimension xs.size) := by
  unfold observe?
  simp [obsOfArray?, h]

theorem observe?_preserves_layout
    (state : StreamingContract)
    (xs : Array Real)
    (next : StreamingContract)
    (h : observe? state xs = .ok next) :
    next.layout = state.layout := by
  unfold observe? at h
  cases hobs : obsOfArray? state.dimension xs <;> simp [hobs] at h
  cases h
  rfl

theorem observe?_increments_sampleCount
    (state : StreamingContract)
    (xs : Array Real)
    (next : StreamingContract)
    (h : observe? state xs = .ok next) :
    next.sampleCount = state.sampleCount + 1 := by
  unfold observe? at h
  cases hobs : obsOfArray? state.dimension xs <;> simp [hobs] at h
  cases h
  simp [sampleCount, observe_n_eq_succ]

end UpdateFacts

section ExtractionFacts

theorem covariance?_eq_exact
    (state : StreamingContract)
    (h : 2 ≤ state.sampleCount) :
    covariance? state = .ok (covarianceMatrix state.estimator) := by
  unfold covariance? sampleCount
  split
  · rfl
  · contradiction

theorem ledoitWolf?_fixedAlpha_eq
    (state : StreamingContract)
    (α : Real)
    (h : 2 ≤ state.sampleCount) :
    ledoitWolf? state (.fixedAlpha α) = .ok (ledoitWolfShrink α (covarianceMatrix state.estimator)) := by
  unfold ledoitWolf?
  rw [covariance?_eq_exact state h]
  simp

theorem ledoitWolf?_clippedAlpha_eq
    (state : StreamingContract)
    (α : Real)
    (h : 2 ≤ state.sampleCount) :
    ledoitWolf? state (.clippedAlpha α)
      = .ok (boundedLedoitWolfShrink α (covarianceMatrix state.estimator)) := by
  unfold ledoitWolf?
  rw [covariance?_eq_exact state h]
  simp

theorem covariance?_eq_insufficientSamples
    (state : StreamingContract)
    (h : state.sampleCount < 2) :
    covariance? state = .error (.insufficientSamples state.sampleCount) := by
  unfold covariance?
  by_cases hEnough : 2 ≤ state.sampleCount
  · exfalso
    exact (not_le_of_gt h) hEnough
  · simp [hEnough]

theorem ledoitWolf?_eq_insufficientSamples
    (state : StreamingContract)
    (mode : ShrinkageMode)
    (h : state.sampleCount < 2) :
    ledoitWolf? state mode = .error (.insufficientSamples state.sampleCount) := by
  unfold ledoitWolf?
  rw [covariance?_eq_insufficientSamples state h]
  rfl

end ExtractionFacts

section BufferFacts

theorem encodeRowMajor_length {k : Nat} (S : CovMatrix k) :
    (encodeRowMajor S).size = MatrixLayout.outputSize .rowMajor k := by
  simp [encodeRowMajor, rowMajorEntries_length]

theorem encodeUpperTrianglePacked_length {k : Nat} (S : CovMatrix k) :
    (encodeUpperTrianglePacked S).size = MatrixLayout.outputSize .upperTrianglePacked k := by
  simp [encodeUpperTrianglePacked, upperTriangleEntries_length]

theorem encodeMatrix_length (layout : MatrixLayout) {k : Nat} (S : CovMatrix k) :
    (encodeMatrix layout S).size = layout.outputSize k := by
  cases layout <;> simp [encodeMatrix, encodeRowMajor_length, encodeUpperTrianglePacked_length]

/--
Main contract theorem of this file.

Successful checked covariance extraction returns the exact mathematical object,
encoded in the layout selected by the runtime-facing API.
-/

theorem covarianceBuffer?_eq_exact
    (state : StreamingContract)
    (h : 2 ≤ state.sampleCount) :
    covarianceBuffer? state
      = .ok (encodeMatrix state.layout (covarianceMatrix state.estimator)) := by
  unfold covarianceBuffer?
  rw [covariance?_eq_exact state h]
  simp

theorem ledoitWolfBuffer?_fixedAlpha_eq
    (state : StreamingContract)
    (α : Real)
    (h : 2 ≤ state.sampleCount) :
    ledoitWolfBuffer? state (.fixedAlpha α)
      = .ok (encodeMatrix state.layout (ledoitWolfShrink α (covarianceMatrix state.estimator))) := by
  unfold ledoitWolfBuffer?
  rw [ledoitWolf?_fixedAlpha_eq state α h]
  simp

theorem ledoitWolfBuffer?_clippedAlpha_eq
    (state : StreamingContract)
    (α : Real)
    (h : 2 ≤ state.sampleCount) :
    ledoitWolfBuffer? state (.clippedAlpha α)
      = .ok (encodeMatrix state.layout (boundedLedoitWolfShrink α (covarianceMatrix state.estimator))) := by
  unfold ledoitWolfBuffer?
  rw [ledoitWolf?_clippedAlpha_eq state α h]
  simp

theorem covarianceBuffer?_length
    (state : StreamingContract) :
    (encodeMatrix state.layout (covarianceMatrix state.estimator)).size
      = state.layout.outputSize state.dimension := by
  simpa using encodeMatrix_length state.layout (covarianceMatrix state.estimator)

theorem ledoitWolfBuffer?_fixedAlpha_length
    (state : StreamingContract)
    (α : Real) :
    (encodeMatrix state.layout (ledoitWolfShrink α (covarianceMatrix state.estimator))).size
      = state.layout.outputSize state.dimension := by
  simpa using encodeMatrix_length state.layout
    (ledoitWolfShrink α (covarianceMatrix state.estimator))

theorem ledoitWolfBuffer?_clippedAlpha_length
    (state : StreamingContract)
    (α : Real) :
    (encodeMatrix state.layout (boundedLedoitWolfShrink α (covarianceMatrix state.estimator))).size
      = state.layout.outputSize state.dimension := by
  simpa using encodeMatrix_length state.layout
    (boundedLedoitWolfShrink α (covarianceMatrix state.estimator))

end BufferFacts

end ApiFacts

end StreamingContract

end CheckedState

end Covstream
