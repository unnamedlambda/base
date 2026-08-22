import MlSurface

/-!
  # What the gpt-oss-20b scanners are allowed to trust

  This is one definition in a module of its own, and the reason is mechanical:
  each artifact generator is an executable with its own `main`, so no single
  module can import two of them. The application has two generators — the
  per-piece artifacts in `GptOssAlgorithm` and the whole model in
  `GptOssDecode` — and therefore two scanners. Both need the same surface, and
  a surface duplicated in two places is a surface that will differ in one.
-/

/-- **This application's surface: the model stack's, plus one rendering.**

    A guard that measures a PTX text has the text in its closure, and a text
    with a float literal in it was rendered by `Float.toString`. The model
    stack's surface already admits `Float32.toString` for exactly this reason;
    the MXFP4 kernels' immediates go through the double-precision one, so it is
    named here rather than added to `mlSurface`, where it would silently widen
    the surface of five pipelines that do not need it.

    What it is trusted for is narrow: the guards say a *rendered* text fits its
    slot. Whether the digits are the right digits is not a claim any of them
    makes, and the emitted kernels are checked bit-exact elsewhere. -/
def gptOssSurface : TrustScan.Surface :=
  { TrustScan.mlSurface with
    allowedOpaque := `Float.toString :: TrustScan.mlSurface.allowedOpaque }
