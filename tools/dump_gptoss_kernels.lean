import GptOssKernels
/-- The MXFP4 kernels as one module, for `applications/gpt-oss/kernel_test.py`.

    Not a `lean_exe` in the algorithms package: every generator there is
    expected to produce an artifact, and this produces a test input. -/
def main : IO Unit := IO.println GptOssKernels.moduleText
