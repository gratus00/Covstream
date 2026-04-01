import Mathlib
namespace Covstream

structure WelfordCov (k : Nat) where
  n                 : Nat             -- number of observations seen so far
  mean              : Fin k -> Real  -- current running mean for each asset
  C        : Fin k -> Fin k -> Real  -- accumulator for covariance (not yet divided)


def WelfordCov.empty (k : Nat) : WelfordCov k :=
{
  n := 0
  mean := fun _ => 0
  C:= fun _ _ => 0 }


noncomputable def WelfordCov.observe( est : WelfordCov k ) ( x: Fin k -> Real ) : WelfordCov k :=
  let n'    := est.n +1
  let nf    : Real := n'
  let delta := fun i => x i - est.mean i
  let mean' := fun i => est.mean i + delta i / nf
  let C'    := fun i j => est.C i j + delta i * ( x j - mean' j)
  { n       := n'
    mean    := mean'
    C       := C' }

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

theorem observe_C_symm { k : Nat} (est : WelfordCov k)
    (x : Fin k -> Real)
    (h: ∀ i j, est.C i j = est.C j i)
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


end Covstream
