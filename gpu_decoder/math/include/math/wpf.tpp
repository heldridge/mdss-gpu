#pragma once

#include <chrono>

#include "math/execution.h"
#include "math/ff_math.h"

using Clock = std::chrono::high_resolution_clock;

template <typename T>
__host__ __device__ inline void push_stack(T* stack, int& top, T value) {
  stack[top++] = value;
}

template <typename T>
__host__ __device__ inline T pop_stack(T* stack, int& top) {
  top--;
  return stack[top];
}

template <typename Field>
__device__ uint32_t calculate_zeroing_coeff(uint32_t higher_lc,
                                            uint32_t lower_lc) {
  if (lower_lc == 0) return 0;

  return Field::mul(Field::inverse(lower_lc), Field::add_inverse(higher_lc));
}

template <typename Field>
__global__ void _set_current_ero(WPFData d, EROConfig* cERO) {
  int c_top = d.d_conflicts_stack_top[0];

  if (c_top != 0) {
    int col_with_conflict =
        pop_stack(d.d_conflicts_stack, *d.d_conflicts_stack_top);

    gpuPivotInfo gpi_a = pop_stack(
        d.d_rows_with_pivot + (col_with_conflict * d.d_arr_meta.num_rows),
        d.d_rwp_tops[col_with_conflict]);
    gpuPivotInfo gpi_b = pop_stack(
        d.d_rows_with_pivot + (col_with_conflict * d.d_arr_meta.num_rows),
        d.d_rwp_tops[col_with_conflict]);

    gpuPivotInfo higher;
    gpuPivotInfo lower;

    if (gpi_a.degree >= gpi_b.degree) {
      higher = gpi_a;
      lower = gpi_b;
    } else {
      higher = gpi_b;
      lower = gpi_a;
    }

    push_stack(
        d.d_rows_with_pivot + (col_with_conflict * d.d_arr_meta.num_rows),
        d.d_rwp_tops[col_with_conflict], lower);
    uint32_t higher_lc =
        d.d_arr[(higher.row_idx * d.d_arr_meta.num_cols *
                 d.d_arr_meta.max_degree) +
                (col_with_conflict * d.d_arr_meta.max_degree) + higher.degree];

    uint32_t lower_lc =
        d.d_arr[(lower.row_idx * d.d_arr_meta.num_cols *
                 d.d_arr_meta.max_degree) +
                (col_with_conflict * d.d_arr_meta.max_degree) + lower.degree];
    uint32_t zeroing_coeff =
        calculate_zeroing_coeff<Field>(higher_lc, lower_lc);
    cERO[0] = {higher.row_idx, lower.row_idx, zeroing_coeff,
               higher.degree - lower.degree};
  } else {
    cERO[0] = {0, 0, 0, -1};
  }
}

template <typename Field>
void FFMath<Field>::set_current_ero(WPFData d, EROConfig* cERO,
                                    cudaStream_t stream) {
  _set_current_ero<Field><<<1, 1, 0, stream>>>(d, cERO);
}

template <typename Field>
__global__ void _perform_current_ero(uint32_t* d_arr, ArrayMetadata d_arr_meta,
                                     EROConfig* cERO) {
  EROConfig ero = cERO[0];

  if (ero.shift < 0) {
    return;
  }

  int z = (blockIdx.x * blockDim.x) + threadIdx.x;
  int col = blockIdx.y;

  if (col < d_arr_meta.num_cols && (z + ero.shift < d_arr_meta.max_degree)) {
    int target_idx =
        (ero.target_row_idx * d_arr_meta.num_cols * d_arr_meta.max_degree) +
        (col * d_arr_meta.max_degree) + z + ero.shift;
    int source_idx =
        (ero.source_row_idx * d_arr_meta.num_cols * d_arr_meta.max_degree) +
        (col * d_arr_meta.max_degree) + z;

    uint32_t result =
        Field::add(d_arr[target_idx],
                   Field::mul(d_arr[source_idx], ero.zeroing_coefficient));
    d_arr[target_idx] = result;
  }
}

template <typename Field>
void FFMath<Field>::perform_current_ero(uint32_t* d_arr,
                                        ArrayMetadata d_arr_meta,
                                        EROConfig* cERO, cudaStream_t stream) {
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(
      (d_arr_meta.max_degree + threadsPerBlock.x - 1) / threadsPerBlock.x,
      d_arr_meta.num_cols);

  _perform_current_ero<Field>
      <<<numBlocks, threadsPerBlock, 0, stream>>>(d_arr, d_arr_meta, cERO);
}

template <typename Field>
WPFResult FFMath<Field>::weak_popov_form(uint32_t* d_arr,
                                         ArrayMetadata d_arr_meta,
                                         int expected_steps) {
  auto t0 = Clock::now();

  auto data = init_wpf_data_structures(d_arr, d_arr_meta);

  EROConfig* cERO;
  cudaMalloc(&cERO, sizeof(EROConfig));

  int* d_step;
  cudaMalloc(&d_step, sizeof(int));
  cudaMemset(d_step, 0, sizeof(int));

  init_metadata(data);
  auto t1 = Clock::now();

  cudaGraph_t graph;
  cudaGraphCreate(&graph, 0);

  cudaGraphConditionalHandle handle;
  cudaGraphConditionalHandleCreate(&handle, graph, 1,
                                   cudaGraphCondAssignDefault);
  cudaGraphNodeParams cParams = {cudaGraphNodeTypeConditional};
  cParams.conditional.handle = handle;
  cParams.conditional.type = cudaGraphCondTypeWhile;
  cParams.conditional.size = 1;

  cudaGraphNode_t condNode;
  cudaGraphAddNode(&condNode, graph, nullptr, nullptr, 0, &cParams);
  cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  cudaStream_t captureStream;
  cudaStreamCreate(&captureStream);
  cudaStreamBeginCaptureToGraph(captureStream, bodyGraph, nullptr, nullptr, 0,
                                cudaStreamCaptureModeGlobal);

  set_current_ero(data, cERO, captureStream);
  perform_current_ero(data.d_arr, data.d_arr_meta, cERO, captureStream);
  update_metadata_post_ero(data, cERO, captureStream);
  push_best_pivot(data, cERO, captureStream);
  _check_continue<<<1, 1, 0, captureStream>>>(cERO, d_step, expected_steps,
                                              handle);
  cudaGraph_t discardedHandleGraph;
  cudaStreamEndCapture(captureStream, &discardedHandleGraph);

  cudaGraphExec_t graphExec;
  cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto t2 = Clock::now();

  cudaGraphLaunch(graphExec, stream);
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
  auto t3 = Clock::now();

  int h_step[1] = {};
  cudaMemcpy(h_step, d_step, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(data.d_rows_with_pivot);
  cudaFree(data.d_rwp_tops);
  cudaFree(data.d_conflicts_stack);
  cudaFree(data.d_conflicts_stack_top);
  cudaFree(data.d_col_best_degree);
  cudaFree(d_step);
  cudaFree(cERO);
  cudaStreamDestroy(captureStream);

  WPFTiming wpft = {.initial_allocation = t1 - t0,
                    .graph_setup = t2 - t1,
                    .main_loop = t3 - t2};

  WPFResult wpfr = {.timing = wpft, .steps_taken = h_step[0]};

  return wpfr;
}
