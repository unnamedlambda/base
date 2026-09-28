import Lake
open Lake DSL

require common from "../common"
-- For the `artifacts` target only; nothing here imports it.
require algorithmLib from "../../../../../lean/lib"

package alpha

lean_lib Alpha

lean_exe genalpha where
  root := `Alpha.Gen
