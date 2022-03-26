
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

// Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_nn_align2
using Operation_cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_nn_align2 =
    cutlass::gemm::device::GemmUniversal<
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 64, 32>,
        cutlass::gemm::GemmShape<64, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,

        cutlass::epilogue::thread::
            LinearCombination<cutlass::half_t, 2, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        6,
        2,
        2,

        cutlass::arch::OpMultiplyAdd

        >;

using BMMInstance =
    Operation_cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_nn_align2;

void bmm(cutlass::half_t* a_ptr,
         cutlass::half_t* b_ptr,
         cutlass::half_t* c_ptr,
         int* a_dim0,
         int* a_dim1,
         int* a_dim2,
         int* b_dim0,
         int* b_dim1,
         int* b_dim2,
         int* c_dim0,
         int* c_dim1,
         int* c_dim2) {
  int AB = *a_dim0;
  int AK = *a_dim1;
  int AM = *a_dim2;
  int BB = *b_dim0;
  int BN = *b_dim1;
  int BK = *b_dim2;
  int CB = AB;
  int CM = AM;
  int CN = BN;
  *c_dim0 = CB;
  *c_dim1 = CM;
  *c_dim2 = CN;
  const int B = AB;
  const int M = AM;
  const int N = BN;
  const int K = AK;

  if (B == 1024 && M == 128 && N == 30 && K == 752) {
    //  TODO: cast to right dtype
    using ElementComputeEpilogue = typename BMMInstance::ElementAccumulator;

    typename BMMInstance::Arguments arguments{

        cutlass::gemm::GemmUniversalMode::kBatched,
        {AM, BN, AK},
        AB,
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
        (void*)a_ptr,
        (void*)b_ptr,
        (void*)c_ptr,
        (void*)c_ptr,
        AK * AM,
        0,
        CM * CN,
        CM * CN,
        AM,
        BK,
        CN,
        CN

    };
    BMMInstance gemm_op;
    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    // TODO: handle workspace correctly
    // This is a shitty code happened to be ok for non-split-k case
    // Because for fprop the workspace is 0
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    status = gemm_op();
    CUTLASS_CHECK(status);
    return;
  }
  throw std::runtime_error(
      "Unsupported workload for this conv2d specialization.");
}
int main(int argc, char** argv) {
  int B = std::atoi(argv[1]);
  int M = std::atoi(argv[2]);
  int N = std::atoi(argv[3]);
  int K = std::atoi(argv[4]);
  // This is a special bmm problem
  // X is [1, M, K]
  // W is [B. N, K]
  // Bias is [B, N]
  // Y is [B, M, N]

  int a_dim0 = B;
  int a_dim1 = K;
  int a_dim2 = M;
  int b_dim0 = 1;
  int b_dim1 = N;
  int b_dim2 = K;
  int c_dim0 = B;
  int c_dim1 = M;
  int c_dim2 = N;

  int AB = a_dim0;
  int AM = a_dim1;
  int AK = a_dim2;
  int BB = b_dim0;
  int BN = b_dim1;
  int BK = a_dim2;
  int CB = AB;
  int CM = AM;
  int CN = BN;

  using ElementOutput = typename GemmInstance::ElementC;
  using ElementInputA = typename GemmInstance::ElementA;
  using ElementInputB = typename GemmInstance::ElementB;

  cutlass::HostTensor<ElementInputA, typename GemmInstance::LayoutA> a(
      {a_dim0 * a_dim1, a_dim2});
  cutlass::HostTensor<ElementInputB, typename GemmInstance::LayoutB> b(
      {b_dim0 * b_dim1, b_dim2});
  cutlass::HostTensor<ElementOutput, typename GemmInstance::LayoutC> c(
      {c_dim0 * c_dim1, c_dim2});
  cutlass::HostTensor<ElementInputA, typename GemmInstance::LayoutA> bias(
      {c_dim2});

  // warmup

  bmm((cutlass::half_t*)a.device_data(), (cutlass::half_t*)b.device_data(),
      (cutlass::half_t*)bias.device_data(), (cutlass::half_t*)c.device_data(),
      &a_dim0, &a_dim1, &a_dim2, &b_dim0, &b_dim1, &b_dim2, &c_dim0, &c_dim1,
      &c_dim2);
  cudaEvent_t events[2];
  for (auto& event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0]);
  for (int i = 0; i < 5; ++i) {
    bmm((cutlass::half_t*)a.device_data(), (cutlass::half_t*)b.device_data(),
        (cutlass::half_t*)bias.device_data(), (cutlass::half_t*)c.device_data(),
        &a_dim0, &a_dim1, &a_dim2, &b_dim0, &b_dim1, &b_dim2, &c_dim0, &c_dim1,
        &c_dim2);
  }
  cudaEventRecord(events[1]);
  cudaEventSynchronize(events[1]);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }

  std::cout << runtime_ms << std::endl;
}