import Lake
open Lake DSL

package covstream where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.29.0"

@[default_target]
lean_lib Covstream where

lean_exe covstream where
  root := `Main
