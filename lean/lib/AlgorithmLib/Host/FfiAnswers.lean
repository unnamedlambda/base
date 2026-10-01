module
public import AlgorithmLib.Host.Lifecycle
meta import AlgorithmLib.Host.Lifecycle
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Every entry point answers a scalar

What an entry point answers, when it answers, is one scalar value
(`callBits_scalar`), so a program that passes an answer on to the next call
passes known bits, whatever the answer is.
-/

namespace AlgorithmLib.HProg.Contracts

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

set_option maxHeartbeats 1000000 in
theorem scalar_fileRead {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .fileRead bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.fileRead).result.getD .i64) x := by
  change ffiFileRead bits w = some (some v, w') at h
  unfold ffiFileRead at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_fileWrite {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .fileWrite bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.fileWrite).result.getD .i64) x := by
  change ffiFileWrite bits w = some (some v, w') at h
  unfold ffiFileWrite at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_fileReadToPtr {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .fileReadToPtr bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.fileReadToPtr).result.getD .i64) x := by
  change ffiFileReadToPtr bits w = some (some v, w') at h
  unfold ffiFileReadToPtr at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_fileWriteFromPtr {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .fileWriteFromPtr bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.fileWriteFromPtr).result.getD .i64) x := by
  change ffiFileWriteFromPtr bits w = some (some v, w') at h
  unfold ffiFileWriteFromPtr at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_stdinReadline {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .stdinReadline bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.stdinReadline).result.getD .i64) x := by
  change ffiStdinReadline bits w = some (some v, w') at h
  unfold ffiStdinReadline at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_stdoutWrite {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .stdoutWrite bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.stdoutWrite).result.getD .i64) x := by
  change ffiStdoutWrite bits w = some (some v, w') at h
  unfold ffiStdoutWrite at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_sinf {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .sinf bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.sinf).result.getD .i64) x := by
  change ffiSinf bits w = some (some v, w') at h
  unfold ffiSinf at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cosf {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cosf bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cosf).result.getD .i64) x := by
  change ffiCosf bits w = some (some v, w') at h
  unfold ffiCosf at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_powf {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .powf bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.powf).result.getD .i64) x := by
  change ffiPowf bits w = some (some v, w') at h
  unfold ffiPowf at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_htInit {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .htInit bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.htInit).result.getD .i64) x := by
  change ffiHtInit bits w = some (some v, w') at h
  unfold ffiHtInit at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_htCleanup {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .htCleanup bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.htCleanup).result.getD .i64) x := by
  change ffiHtCleanup bits w = some (some v, w') at h
  unfold ffiHtCleanup at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_htCreate {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .htCreate bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.htCreate).result.getD .i64) x := by
  change ffiHtCreate bits w = some (some v, w') at h
  unfold ffiHtCreate at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_htCount {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .htCount bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.htCount).result.getD .i64) x := by
  change ffiHtCount bits w = some (some v, w') at h
  unfold ffiHtCount at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_htLookup {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .htLookup bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.htLookup).result.getD .i64) x := by
  change ffiHtLookup bits w = some (some v, w') at h
  unfold ffiHtLookup at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_htInsert {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .htInsert bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.htInsert).result.getD .i64) x := by
  change ffiHtInsert bits w = some (some v, w') at h
  unfold ffiHtInsert at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_htIncrement {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .htIncrement bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.htIncrement).result.getD .i64) x := by
  change ffiHtIncrement bits w = some (some v, w') at h
  unfold ffiHtIncrement at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_htGetEntry {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .htGetEntry bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.htGetEntry).result.getD .i64) x := by
  change ffiHtGetEntry bits w = some (some v, w') at h
  unfold ffiHtGetEntry at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaInit {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaInit bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaInit).result.getD .i64) x := by
  change ffiCudaInit bits w = some (some v, w') at h
  unfold ffiCudaInit at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaCleanup {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaCleanup bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaCleanup).result.getD .i64) x := by
  change ffiCudaCleanup bits w = some (some v, w') at h
  unfold ffiCudaCleanup at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaCreateBuffer {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaCreateBuffer bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaCreateBuffer).result.getD .i64) x := by
  change ffiCudaCreateBuffer bits w = some (some v, w') at h
  unfold ffiCudaCreateBuffer at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaUpload {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaUpload bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaUpload).result.getD .i64) x := by
  change ffiCudaUpload bits w = some (some v, w') at h
  unfold ffiCudaUpload at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaUploadOffset {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaUploadOffset bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaUploadOffset).result.getD .i64) x := by
  change ffiCudaUploadOffset bits w = some (some v, w') at h
  unfold ffiCudaUploadOffset at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaDownload {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaDownload bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaDownload).result.getD .i64) x := by
  change ffiCudaDownload bits w = some (some v, w') at h
  unfold ffiCudaDownload at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaDownloadOffset {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaDownloadOffset bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaDownloadOffset).result.getD .i64) x := by
  change ffiCudaDownloadOffset bits w = some (some v, w') at h
  unfold ffiCudaDownloadOffset at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaFreeBuffer {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaFreeBuffer bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaFreeBuffer).result.getD .i64) x := by
  change ffiCudaFreeBuffer bits w = some (some v, w') at h
  unfold ffiCudaFreeBuffer at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaSync {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaSync bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaSync).result.getD .i64) x := by
  change ffiCudaSync bits w = some (some v, w') at h
  unfold ffiCudaSync at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaLaunch {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaLaunch bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaLaunch).result.getD .i64) x := by
  change ffiCudaLaunch bits w = some (some v, w') at h
  unfold ffiCudaLaunch at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cublasSgemv {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cublasSgemv bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cublasSgemv).result.getD .i64) x := by
  change ffiCublasSgemv bits w = some (some v, w') at h
  unfold ffiCublasSgemv at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cublasSgemm {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cublasSgemm bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cublasSgemm).result.getD .i64) x := by
  change ffiCublasSgemm bits w = some (some v, w') at h
  unfold ffiCublasSgemm at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cublasGemmExBf16 {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cublasGemmExBf16 bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cublasGemmExBf16).result.getD .i64) x := by
  change ffiCublasGemmExBf16 bits w = some (some v, w') at h
  unfold ffiCublasGemmExBf16 at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cublasSgemvOnStream {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cublasSgemvOnStream bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cublasSgemvOnStream).result.getD .i64) x := by
  change ffiCublasSgemvOnStream bits w = some (some v, w') at h
  unfold ffiCublasSgemvOnStream at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cublasGemmStridedBatchedExBf16 {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cublasGemmStridedBatchedExBf16 bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cublasGemmStridedBatchedExBf16).result.getD .i64) x := by
  change ffiCublasGemmStridedBatchedExBf16 bits w = some (some v, w') at h
  unfold ffiCublasGemmStridedBatchedExBf16 at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cublasPtrArray {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cublasPtrArray bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cublasPtrArray).result.getD .i64) x := by
  change ffiCublasPtrArray bits w = some (some v, w') at h
  unfold ffiCublasPtrArray at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cublasSgemmBatchedOnStream {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cublasSgemmBatchedOnStream bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cublasSgemmBatchedOnStream).result.getD .i64) x := by
  change ffiCublasSgemmBatchedOnStream bits w = some (some v, w') at h
  unfold ffiCublasSgemmBatchedOnStream at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaLaunchNamedOnStream {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaLaunchNamedOnStream bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaLaunchNamedOnStream).result.getD .i64) x := by
  change ffiCudaLaunchNamedOnStream bits w = some (some v, w') at h
  unfold ffiCudaLaunchNamedOnStream at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaEventElapsedMsBits {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaEventElapsedMsBits bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaEventElapsedMsBits).result.getD .i64) x := by
  change ffiCudaEventElapsedMsBits bits w = some (some v, w') at h
  unfold ffiCudaEventElapsedMsBits at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaUploadAsync {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaUploadAsync bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaUploadAsync).result.getD .i64) x := by
  change ffiCudaUploadAsync bits w = some (some v, w') at h
  unfold ffiCudaUploadAsync at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaUploadOffsetAsync {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaUploadOffsetAsync bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaUploadOffsetAsync).result.getD .i64) x := by
  change ffiCudaUploadOffsetAsync bits w = some (some v, w') at h
  unfold ffiCudaUploadOffsetAsync at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaDownloadAsync {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaDownloadAsync bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaDownloadAsync).result.getD .i64) x := by
  change ffiCudaDownloadAsync bits w = some (some v, w') at h
  unfold ffiCudaDownloadAsync at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cublasSgemmOnStream {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cublasSgemmOnStream bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cublasSgemmOnStream).result.getD .i64) x := by
  change ffiCublasSgemmOnStream bits w = some (some v, w') at h
  unfold ffiCublasSgemmOnStream at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaLaunchOnStream {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaLaunchOnStream bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaLaunchOnStream).result.getD .i64) x := by
  change ffiCudaLaunchOnStream bits w = some (some v, w') at h
  unfold ffiCudaLaunchOnStream at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaStreamCreate {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaStreamCreate bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaStreamCreate).result.getD .i64) x := by
  change ffiCudaStreamCreate bits w = some (some v, w') at h
  unfold ffiCudaStreamCreate at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaStreamSync {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaStreamSync bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaStreamSync).result.getD .i64) x := by
  change ffiCudaStreamSync bits w = some (some v, w') at h
  unfold ffiCudaStreamSync at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaStreamDestroy {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaStreamDestroy bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaStreamDestroy).result.getD .i64) x := by
  change ffiCudaStreamDestroy bits w = some (some v, w') at h
  unfold ffiCudaStreamDestroy at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaEventCreate {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaEventCreate bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaEventCreate).result.getD .i64) x := by
  change ffiCudaEventCreate bits w = some (some v, w') at h
  unfold ffiCudaEventCreate at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaEventRecord {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaEventRecord bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaEventRecord).result.getD .i64) x := by
  change ffiCudaEventRecord bits w = some (some v, w') at h
  unfold ffiCudaEventRecord at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaStreamWaitEvent {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaStreamWaitEvent bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaStreamWaitEvent).result.getD .i64) x := by
  change ffiCudaStreamWaitEvent bits w = some (some v, w') at h
  unfold ffiCudaStreamWaitEvent at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaEventDestroy {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaEventDestroy bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaEventDestroy).result.getD .i64) x := by
  change ffiCudaEventDestroy bits w = some (some v, w') at h
  unfold ffiCudaEventDestroy at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaGraphBeginCapture {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaGraphBeginCapture bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaGraphBeginCapture).result.getD .i64) x := by
  change ffiCudaGraphBeginCapture bits w = some (some v, w') at h
  unfold ffiCudaGraphBeginCapture at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaGraphEndCapture {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaGraphEndCapture bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaGraphEndCapture).result.getD .i64) x := by
  change ffiCudaGraphEndCapture bits w = some (some v, w') at h
  unfold ffiCudaGraphEndCapture at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaGraphUpload {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaGraphUpload bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaGraphUpload).result.getD .i64) x := by
  change ffiCudaGraphUpload bits w = some (some v, w') at h
  unfold ffiCudaGraphUpload at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaGraphLaunch {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaGraphLaunch bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaGraphLaunch).result.getD .i64) x := by
  change ffiCudaGraphLaunch bits w = some (some v, w') at h
  unfold ffiCudaGraphLaunch at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaGraphDestroy {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaGraphDestroy bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaGraphDestroy).result.getD .i64) x := by
  change ffiCudaGraphDestroy bits w = some (some v, w') at h
  unfold ffiCudaGraphDestroy at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaLaunchNamed {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaLaunchNamed bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaLaunchNamed).result.getD .i64) x := by
  change ffiCudaLaunchNamed bits w = some (some v, w') at h
  unfold ffiCudaLaunchNamed at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaPinnedAlloc {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaPinnedAlloc bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaPinnedAlloc).result.getD .i64) x := by
  change ffiCudaPinnedAlloc bits w = some (some v, w') at h
  unfold ffiCudaPinnedAlloc at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaPinnedPtr {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaPinnedPtr bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaPinnedPtr).result.getD .i64) x := by
  change ffiCudaPinnedPtr bits w = some (some v, w') at h
  unfold ffiCudaPinnedPtr at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaPinnedPtrAt {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaPinnedPtrAt bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaPinnedPtrAt).result.getD .i64) x := by
  change ffiCudaPinnedPtrAt bits w = some (some v, w') at h
  unfold ffiCudaPinnedPtrAt at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaPinnedFree {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaPinnedFree bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaPinnedFree).result.getD .i64) x := by
  change ffiCudaPinnedFree bits w = some (some v, w') at h
  unfold ffiCudaPinnedFree at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaMemInfoFree {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaMemInfoFree bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaMemInfoFree).result.getD .i64) x := by
  change ffiCudaMemInfoFree bits w = some (some v, w') at h
  unfold ffiCudaMemInfoFree at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cudaMemInfoTotal {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cudaMemInfoTotal bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cudaMemInfoTotal).result.getD .i64) x := by
  change ffiCudaMemInfoTotal bits w = some (some v, w') at h
  unfold ffiCudaMemInfoTotal at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_gpuInit {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .gpuInit bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.gpuInit).result.getD .i64) x := by
  change ffiGpuInit bits w = some (some v, w') at h
  unfold ffiGpuInit at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_gpuCleanup {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .gpuCleanup bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.gpuCleanup).result.getD .i64) x := by
  change ffiGpuCleanup bits w = some (some v, w') at h
  unfold ffiGpuCleanup at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_gpuCreateBuffer {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .gpuCreateBuffer bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.gpuCreateBuffer).result.getD .i64) x := by
  change ffiGpuCreateBuffer bits w = some (some v, w') at h
  unfold ffiGpuCreateBuffer at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_gpuCreatePipeline {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .gpuCreatePipeline bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.gpuCreatePipeline).result.getD .i64) x := by
  change ffiGpuCreatePipeline bits w = some (some v, w') at h
  unfold ffiGpuCreatePipeline at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_gpuUpload {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .gpuUpload bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.gpuUpload).result.getD .i64) x := by
  change ffiGpuUpload bits w = some (some v, w') at h
  unfold ffiGpuUpload at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_gpuUploadPtr {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .gpuUploadPtr bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.gpuUploadPtr).result.getD .i64) x := by
  change ffiGpuUploadPtr bits w = some (some v, w') at h
  unfold ffiGpuUploadPtr at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_gpuDispatch {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .gpuDispatch bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.gpuDispatch).result.getD .i64) x := by
  change ffiGpuDispatch bits w = some (some v, w') at h
  unfold ffiGpuDispatch at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_gpuDownload {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .gpuDownload bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.gpuDownload).result.getD .i64) x := by
  change ffiGpuDownload bits w = some (some v, w') at h
  unfold ffiGpuDownload at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_gpuDownloadPtr {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .gpuDownloadPtr bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.gpuDownloadPtr).result.getD .i64) x := by
  change ffiGpuDownloadPtr bits w = some (some v, w') at h
  unfold ffiGpuDownloadPtr at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_lmdbInit {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .lmdbInit bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.lmdbInit).result.getD .i64) x := by
  change ffiLmdbInit bits w = some (some v, w') at h
  unfold ffiLmdbInit at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_lmdbCleanup {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .lmdbCleanup bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.lmdbCleanup).result.getD .i64) x := by
  change ffiLmdbCleanup bits w = some (some v, w') at h
  unfold ffiLmdbCleanup at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_lmdbOpen {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .lmdbOpen bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.lmdbOpen).result.getD .i64) x := by
  change ffiLmdbOpen bits w = some (some v, w') at h
  unfold ffiLmdbOpen at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_lmdbBeginWriteTxn {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .lmdbBeginWriteTxn bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.lmdbBeginWriteTxn).result.getD .i64) x := by
  change ffiLmdbBeginWriteTxn bits w = some (some v, w') at h
  unfold ffiLmdbBeginWriteTxn at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_lmdbPut {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .lmdbPut bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.lmdbPut).result.getD .i64) x := by
  change ffiLmdbPut bits w = some (some v, w') at h
  unfold ffiLmdbPut at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_lmdbCommitWriteTxn {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .lmdbCommitWriteTxn bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.lmdbCommitWriteTxn).result.getD .i64) x := by
  change ffiLmdbCommitWriteTxn bits w = some (some v, w') at h
  unfold ffiLmdbCommitWriteTxn at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_lmdbCursorScan {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .lmdbCursorScan bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.lmdbCursorScan).result.getD .i64) x := by
  change ffiLmdbCursorScan bits w = some (some v, w') at h
  unfold ffiLmdbCursorScan at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_windowInit {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .windowInit bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.windowInit).result.getD .i64) x := by
  change ffiWindowInit bits w = some (some v, w') at h
  unfold ffiWindowInit at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_windowCleanup {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .windowCleanup bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.windowCleanup).result.getD .i64) x := by
  change ffiWindowCleanup bits w = some (some v, w') at h
  unfold ffiWindowCleanup at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_windowOpen {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .windowOpen bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.windowOpen).result.getD .i64) x := by
  change ffiWindowOpen bits w = some (some v, w') at h
  unfold ffiWindowOpen at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_windowPoll {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .windowPoll bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.windowPoll).result.getD .i64) x := by
  change ffiWindowPoll bits w = some (some v, w') at h
  unfold ffiWindowPoll at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_windowPresentGpuBuffer {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .windowPresentGpuBuffer bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.windowPresentGpuBuffer).result.getD .i64) x := by
  change ffiWindowPresentGpuBuffer bits w = some (some v, w') at h
  unfold ffiWindowPresentGpuBuffer at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_nativeLoad {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .nativeLoad bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.nativeLoad).result.getD .i64) x := by
  change ffiNativeLoad bits w = some (some v, w') at h
  unfold ffiNativeLoad at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_threadInit {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .threadInit bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.threadInit).result.getD .i64) x := by
  change ffiThreadInit bits w = some (some v, w') at h
  unfold ffiThreadInit at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_threadCleanup {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .threadCleanup bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.threadCleanup).result.getD .i64) x := by
  change ffiThreadCleanup bits w = some (some v, w') at h
  unfold ffiThreadCleanup at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_threadJoin {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .threadJoin bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.threadJoin).result.getD .i64) x := by
  change ffiThreadJoin bits w = some (some v, w') at h
  unfold ffiThreadJoin at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_nativeFree {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .nativeFree bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.nativeFree).result.getD .i64) x := by
  change ffiNativeFree bits w = some (some v, w') at h
  unfold ffiNativeFree at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_nativeArch {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .nativeArch bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.nativeArch).result.getD .i64) x := by
  change ffiNativeArch bits w = some (some v, w') at h
  unfold ffiNativeArch at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_cpuHas {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .cpuHas bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.cpuHas).result.getD .i64) x := by
  change ffiCpuHas bits w = some (some v, w') at h
  unfold ffiCpuHas at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_fileCreateDirAll {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .fileCreateDirAll bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.fileCreateDirAll).result.getD .i64) x := by
  change ffiFileCreateDirAll bits w = some (some v, w') at h
  unfold ffiFileCreateDirAll at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_memLock {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .memLock bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.memLock).result.getD .i64) x := by
  change ffiGrant "lock" 2 bits w = some (some v, w') at h
  unfold ffiGrant at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_memUnlock {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .memUnlock bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.memUnlock).result.getD .i64) x := by
  change ffiGrant "unlock" 2 bits w = some (some v, w') at h
  unfold ffiGrant at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_memAdviseHuge {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .memAdviseHuge bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.memAdviseHuge).result.getD .i64) x := by
  change ffiGrant "hugepages" 2 bits w = some (some v, w') at h
  unfold ffiGrant at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

set_option maxHeartbeats 1000000 in
theorem scalar_threadPriority {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits .threadPriority bits w = some (some v, w')) :
    ∃ x, v = .sc ((IR.Ffi.threadPriority).result.getD .i64) x := by
  change ffiGrant "priority" 1 bits w = some (some v, w') at h
  unfold ffiGrant at h
  walk h
  all_goals (try (rename_i hv; injection hv with hv; subst hv))
  all_goals exact ⟨_, rfl⟩

/-- **An entry point's answer is a scalar of the type its signature gives.** -/
theorem callBits_scalar (f : IR.Ffi) {bits : List UInt64} {w : World} {v : V} {w' : World}
    (h : callBits f bits w = some (some v, w')) : ∃ x, v = .sc (f.result.getD .i64) x := by
  cases f with
  | fileRead => exact scalar_fileRead h
  | fileWrite => exact scalar_fileWrite h
  | fileReadToPtr => exact scalar_fileReadToPtr h
  | fileWriteFromPtr => exact scalar_fileWriteFromPtr h
  | stdinReadline => exact scalar_stdinReadline h
  | stdoutWrite => exact scalar_stdoutWrite h
  | sinf => exact scalar_sinf h
  | cosf => exact scalar_cosf h
  | powf => exact scalar_powf h
  | htInit => exact scalar_htInit h
  | htCleanup => exact scalar_htCleanup h
  | htCreate => exact scalar_htCreate h
  | htCount => exact scalar_htCount h
  | htLookup => exact scalar_htLookup h
  | htInsert => exact scalar_htInsert h
  | htIncrement => exact scalar_htIncrement h
  | htGetEntry => exact scalar_htGetEntry h
  | cudaInit => exact scalar_cudaInit h
  | cudaCleanup => exact scalar_cudaCleanup h
  | cudaCreateBuffer => exact scalar_cudaCreateBuffer h
  | cudaUpload => exact scalar_cudaUpload h
  | cudaUploadOffset => exact scalar_cudaUploadOffset h
  | cudaDownload => exact scalar_cudaDownload h
  | cudaDownloadOffset => exact scalar_cudaDownloadOffset h
  | cudaFreeBuffer => exact scalar_cudaFreeBuffer h
  | cudaSync => exact scalar_cudaSync h
  | cudaLaunch => exact scalar_cudaLaunch h
  | cublasSgemv => exact scalar_cublasSgemv h
  | cublasSgemm => exact scalar_cublasSgemm h
  | cublasGemmExBf16 => exact scalar_cublasGemmExBf16 h
  | cublasSgemvOnStream => exact scalar_cublasSgemvOnStream h
  | cublasGemmStridedBatchedExBf16 => exact scalar_cublasGemmStridedBatchedExBf16 h
  | cublasPtrArray => exact scalar_cublasPtrArray h
  | cublasSgemmBatchedOnStream => exact scalar_cublasSgemmBatchedOnStream h
  | cudaLaunchNamedOnStream => exact scalar_cudaLaunchNamedOnStream h
  | cudaEventElapsedMsBits => exact scalar_cudaEventElapsedMsBits h
  | cudaUploadAsync => exact scalar_cudaUploadAsync h
  | cudaUploadOffsetAsync => exact scalar_cudaUploadOffsetAsync h
  | cudaDownloadAsync => exact scalar_cudaDownloadAsync h
  | cublasSgemmOnStream => exact scalar_cublasSgemmOnStream h
  | cudaLaunchOnStream => exact scalar_cudaLaunchOnStream h
  | cudaStreamCreate => exact scalar_cudaStreamCreate h
  | cudaStreamSync => exact scalar_cudaStreamSync h
  | cudaStreamDestroy => exact scalar_cudaStreamDestroy h
  | cudaEventCreate => exact scalar_cudaEventCreate h
  | cudaEventRecord => exact scalar_cudaEventRecord h
  | cudaStreamWaitEvent => exact scalar_cudaStreamWaitEvent h
  | cudaEventDestroy => exact scalar_cudaEventDestroy h
  | cudaGraphBeginCapture => exact scalar_cudaGraphBeginCapture h
  | cudaGraphEndCapture => exact scalar_cudaGraphEndCapture h
  | cudaGraphUpload => exact scalar_cudaGraphUpload h
  | cudaGraphLaunch => exact scalar_cudaGraphLaunch h
  | cudaGraphDestroy => exact scalar_cudaGraphDestroy h
  | cudaLaunchNamed => exact scalar_cudaLaunchNamed h
  | cudaPinnedAlloc => exact scalar_cudaPinnedAlloc h
  | cudaPinnedPtr => exact scalar_cudaPinnedPtr h
  | cudaPinnedPtrAt => exact scalar_cudaPinnedPtrAt h
  | cudaPinnedFree => exact scalar_cudaPinnedFree h
  | cudaMemInfoFree => exact scalar_cudaMemInfoFree h
  | cudaMemInfoTotal => exact scalar_cudaMemInfoTotal h
  | gpuInit => exact scalar_gpuInit h
  | gpuCleanup => exact scalar_gpuCleanup h
  | gpuCreateBuffer => exact scalar_gpuCreateBuffer h
  | gpuCreatePipeline => exact scalar_gpuCreatePipeline h
  | gpuUpload => exact scalar_gpuUpload h
  | gpuUploadPtr => exact scalar_gpuUploadPtr h
  | gpuDispatch => exact scalar_gpuDispatch h
  | gpuDownload => exact scalar_gpuDownload h
  | gpuDownloadPtr => exact scalar_gpuDownloadPtr h
  | lmdbInit => exact scalar_lmdbInit h
  | lmdbCleanup => exact scalar_lmdbCleanup h
  | lmdbOpen => exact scalar_lmdbOpen h
  | lmdbBeginWriteTxn => exact scalar_lmdbBeginWriteTxn h
  | lmdbPut => exact scalar_lmdbPut h
  | lmdbCommitWriteTxn => exact scalar_lmdbCommitWriteTxn h
  | lmdbCursorScan => exact scalar_lmdbCursorScan h
  | windowInit => exact scalar_windowInit h
  | windowCleanup => exact scalar_windowCleanup h
  | windowOpen => exact scalar_windowOpen h
  | windowPoll => exact scalar_windowPoll h
  | windowPresentGpuBuffer => exact scalar_windowPresentGpuBuffer h
  | nativeLoad => exact scalar_nativeLoad h
  | threadInit => exact scalar_threadInit h
  | threadCleanup => exact scalar_threadCleanup h
  | threadJoin => exact scalar_threadJoin h
  | nativeFree => exact scalar_nativeFree h
  | nativeArch => exact scalar_nativeArch h
  | cpuHas => exact scalar_cpuHas h
  | fileCreateDirAll => exact scalar_fileCreateDirAll h
  | memLock => exact scalar_memLock h
  | memUnlock => exact scalar_memUnlock h
  | memAdviseHuge => exact scalar_memAdviseHuge h
  | threadPriority => exact scalar_threadPriority h
  | threadSpawn | threadStart | threadFinish => cases h

/-- The same through the call the model makes, for every entry point but
    the thread spawn, which runs a function of the program. -/
theorem callOf_ffi_scalar {f : IR.Ffi} (hf : f ≠ .threadSpawn) {lc : Locals} {vs : List V} {w : World}
    {v : V} {w' : World} (h : callOf lc (.ffi f) vs w = some (some v, w')) :
    ∃ x, v = .sc (f.result.getD .i64) x := by
  have hs : (f == .threadSpawn) = false := by simpa using hf
  simp only [callOf, hs, if_false, Bool.false_eq_true, callImport, ofCname_cname, bind, Option.bind,
    callFfi] at h
  split at h
  · cases h
  · split at h
    · cases h
    · exact callBits_scalar f h

end AlgorithmLib.HProg.Contracts
