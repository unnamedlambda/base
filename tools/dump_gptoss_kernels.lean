import GptOss.Kernels
/-- The gpt-oss kernels as one module, for `applications/gpt-oss/kernel_test.py`
    and `attn_fused_test.py`, which find each kernel by its name.

    Not a `lean_exe` in the algorithms package: every generator there is
    expected to produce an artifact, and this produces a test input. -/
def main : IO Unit :=
  IO.println GptOssKernels.Body.moduleText
