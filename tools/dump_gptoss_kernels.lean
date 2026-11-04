import GptOss.Kernels
/-- The MXFP4 kernels, for `applications/gpt-oss/kernel_test.py`: the module of
    those built as strings, a line `// module`, and the module of those written
    as bodies.

    Not a `lean_exe` in the algorithms package: every generator there is
    expected to produce an artifact, and this produces a test input. -/
def main : IO Unit := do
  IO.println GptOssKernels.moduleText
  IO.println "// module"
  IO.println GptOssKernels.Body.moduleText
