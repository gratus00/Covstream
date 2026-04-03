import Covstream.Welford
import Covstream.LedoitWolf
import Covstream.FrobeniusLoss
import Covstream.Contract
import Covstream.ErrorBounds
import Covstream.Examples

/-!
Top-level umbrella import for the `Covstream` formalization.

The development is organized in six reader-facing modules:

1. `Covstream.Welford`
   The exact Real-valued streaming covariance specification and its correspondence
   with the classical batch covariance formula.
2. `Covstream.LedoitWolf`
   The structural shrinkage layer: target matrix, convex shrinkage, and
   symmetry / PSD preservation.
3. `Covstream.FrobeniusLoss`
   The optimization layer: Frobenius loss expansion and oracle shrinkage
   coefficients.
4. `Covstream.Contract`
   A runtime-facing checked API contract for Rust/C++ implementations.
5. `Covstream.ErrorBounds`
   Abstract perturbation lemmas for future floating-point refinement work.
6. `Covstream.Examples`
   Small concrete 2D examples of the checked API.

The internal optimization code is split between:

* `Covstream.FrobeniusBasic`
* `Covstream.ShrinkageOptimization`
-/
