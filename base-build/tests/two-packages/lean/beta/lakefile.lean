import Lake
open Lake DSL

require common from "../common"
-- For the `artifacts` target only; nothing here imports it.
require algorithmLib from "../../../../../lean/lib"

package beta

lean_lib Beta

lean_exe genbeta where
  root := `Beta.Gen
