import hashlib
import math
from types import ModuleType

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


class Solution:
    def name(self):
        base = "fused_mlp_wmma_A100_vectorized_padded"
        code_hash = hashlib.md5(
            (self.cuda_src_code() + self.cpp_src_code()).encode()
        ).hexdigest()[:8]
        return f"{base}_{code_hash}"

    def cuda_src_code(self):
        return r"""
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// --- Tiling and Block Dimensions ---
// Tuned to increase occupancy on A100. Smaller tiles reduce shared memory
// pressure, allowing more blocks to run concurrently.
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;

// Padding to avoid shared memory bank conflicts.
constexpr int PADDING = 8;
constexpr int TILE_K_PADDED = TILE_K + PADDING;
constexpr int TILE_N_PADDED = TILE_N + PADDING;
// Padding for the output buffer in shared mem to avoid store-related bank conflicts.
constexpr int TILE_N_PADDED_C = TILE_N + PADDING;

// WMMA fragment dimensions for sm_80 (m16n16k16)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Thread block configuration: 4 warps (128 threads) per block.
constexpr int BLOCK_THREADS = 128;
// Warps are arranged in a 2x2 grid. Each warp handles one 16x16 tile.
constexpr int WARPS_PER_BLOCK_M = 2;
constexpr int WARPS_PER_BLOCK_N = 2;
constexpr int WARP_TILE_M = TILE_M / WARPS_PER_BLOCK_M; // 16
constexpr int WARP_TILE_N = TILE_N / WARPS_PER_BLOCK_N; // 16

// --- Shared Memory Size Calculation ---
// Double buffered for A and B to pipeline data loading with computation.
constexpr size_t SMEM_A_BYTES = 2 * TILE_M * TILE_K_PADDED * sizeof(half);      // 2*32*40*2 = 5120 bytes
constexpr size_t SMEM_B_BYTES = 2 * TILE_K * TILE_N_PADDED * sizeof(half);      // 2*32*40*2 = 5120 bytes
// Single buffer for C, padded to avoid bank conflicts on store.
constexpr size_t SMEM_C_BYTES = TILE_M * TILE_N_PADDED_C * sizeof(float);       // 32*40*4 = 5120 bytes
// Buffer for bias vector.
constexpr size_t SMEM_BIAS_BYTES = TILE_N * sizeof(float);                      // 32*4 = 128 bytes
// Total: ~15.5 KB. Allows 10 blocks per SM on A100 (164KB/SM), boosting occupancy to ~62.5%.

__device__ __forceinline__ void load_gmem_tile_to_smem_A_vectorized(
    const float* __restrict__ gmem_ptr, half* smem_ptr, int M, int K, int gmem_m, int gmem_k, int stride) {
    
    constexpr int VEC_SIZE = 4;
    constexpr int LOADS_PER_THREAD = (TILE_M * TILE_K / VEC_SIZE) / BLOCK_THREADS;

    #pragma unroll
    for (int i = 0; i < LOADS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        int m_local = idx / (TILE_K / VEC_SIZE);
        int k_local_base = (idx % (TILE_K / VEC_SIZE)) * VEC_SIZE;

        const int current_gmem_m = gmem_m + m_local;
        const int current_gmem_k = gmem_k + k_local_base;
        
        if (current_gmem_m < M && current_gmem_k < K) {
            float4 val = *reinterpret_cast<const float4*>(&gmem_ptr[current_gmem_m * stride + current_gmem_k]);
            // Convert float4 to 2x half2 and store
            __half2 h1 = __float22half2_rn(make_float2(val.x, val.y));
            __half2 h2 = __float22half2_rn(make_float2(val.z, val.w));
            *reinterpret_cast<__half2*>(&smem_ptr[m_local * TILE_K_PADDED + k_local_base]) = h1;
            *reinterpret_cast<__half2*>(&smem_ptr[m_local * TILE_K_PADDED + k_local_base + 2]) = h2;
        } else {
            // Zero padding for out-of-bounds accesses
            *reinterpret_cast<float2*>(&smem_ptr[m_local * TILE_K_PADDED + k_local_base]) = make_float2(0.0f, 0.0f);
        }
    }
}

__device__ __forceinline__ void load_gmem_tile_to_smem_B_vectorized(
    const float* __restrict__ gmem_ptr, half* smem_ptr, int N, int K, int gmem_n, int gmem_k, int stride) {
    
    constexpr int VEC_SIZE = 4;
    constexpr int LOADS_PER_THREAD = (TILE_K * TILE_N / VEC_SIZE) / BLOCK_THREADS;

    #pragma unroll
    for (int i = 0; i < LOADS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        int k_local = idx / (TILE_N / VEC_SIZE);
        int n_local_base = (idx % (TILE_N / VEC_SIZE)) * VEC_SIZE;
        
        const int current_gmem_k = gmem_k + k_local;
        const int current_gmem_n = gmem_n + n_local_base;

        if (current_gmem_k < K && current_gmem_n < N) {
            float4 val = *reinterpret_cast<const float4*>(&gmem_ptr[current_gmem_k * stride + current_gmem_n]);
            __half2 h1 = __float22half2_rn(make_float2(val.x, val.y));
            __half2 h2 = __float22half2_rn(make_float2(val.z, val.w));
            *reinterpret_cast<__half2*>(&smem_ptr[k_local * TILE_N_PADDED + n_local_base]) = h1;
            *reinterpret_cast<__half2*>(&smem_ptr[k_local * TILE_N_PADDED + n_local_base + 2]) = h2;
        } else {
            *reinterpret_cast<float2*>(&smem_ptr[k_local * TILE_N_PADDED + n_local_base]) = make_float2(0.0f, 0.0f);
        }
    }
}


__global__ void __launch_bounds__(BLOCK_THREADS)
fused_linear_relu_kernel_vectorized_padded(
    const float* __restrict__ X,    // Input matrix (M, K)
    const float* __restrict__ W_T,  // TRANSPOSED Weight matrix (K, N)
    const float* __restrict__ B,    // Bias vector (N)
    float* __restrict__ Y,          // Output matrix (M, N)
    int M, int N, int K,
    int input_row_stride,
    int output_row_stride
) {
    // --- Shared Memory Allocation ---
    extern __shared__ char smem[];
    half* smem_A = reinterpret_cast<half*>(smem);
    half* smem_B = reinterpret_cast<half*>(smem + SMEM_A_BYTES);
    float* smem_C = reinterpret_cast<float*>(smem + SMEM_A_BYTES + SMEM_B_BYTES);
    float* smem_bias = reinterpret_cast<float*>(smem + SMEM_A_BYTES + SMEM_B_BYTES + SMEM_C_BYTES);

    // --- Thread and Block Indexing ---
    const int warp_id = threadIdx.x / 32;
    const int block_m_idx = blockIdx.y;
    const int block_n_idx = blockIdx.x;
    const int warp_row = warp_id / WARPS_PER_BLOCK_N;
    const int warp_col = warp_id % WARPS_PER_BLOCK_N;

    // --- WMMA Fragments ---
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
    wmma::fill_fragment(C_frag, 0.0f);

    // --- Pipelined Main Loop (GEMM) ---
    int current_buffer_idx = 0;
    const int gmem_m_base = block_m_idx * TILE_M;
    const int gmem_n_base = block_n_idx * TILE_N;
    
    // --- Stage 0: Vectorized load of the first K-tile ---
    load_gmem_tile_to_smem_A_vectorized(X, smem_A, M, K, gmem_m_base, 0, input_row_stride);
    load_gmem_tile_to_smem_B_vectorized(W_T, smem_B, N, K, gmem_n_base, 0, N);
    __syncthreads();

    // Loop over the K dimension, pipelining loads and computation
    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += TILE_K) {
        const int next_k_tile_start = k_tile_start + TILE_K;
        const int next_buffer_idx = 1 - current_buffer_idx;

        // Stage 1: Asynchronously load next tile
        if (next_k_tile_start < K) {
            half* next_smem_A = smem_A + (next_buffer_idx * TILE_M * TILE_K_PADDED);
            load_gmem_tile_to_smem_A_vectorized(X, next_smem_A, M, K, gmem_m_base, next_k_tile_start, input_row_stride);
            
            half* next_smem_B = smem_B + (next_buffer_idx * TILE_K * TILE_N_PADDED);
            load_gmem_tile_to_smem_B_vectorized(W_T, next_smem_B, N, K, gmem_n_base, next_k_tile_start, N);
        }

        // Stage 2: Compute on current tile
        const half* current_smem_A = smem_A + (current_buffer_idx * TILE_M * TILE_K_PADDED);
        const half* current_smem_B = smem_B + (current_buffer_idx * TILE_K * TILE_N_PADDED);

        #pragma unroll
        for (int k_step = 0; k_step < TILE_K; k_step += WMMA_K) {
            int smem_A_m_offset = warp_row * WARP_TILE_M;
            int smem_B_n_offset = warp_col * WARP_TILE_N;
            wmma::load_matrix_sync(A_frag, &current_smem_A[smem_A_m_offset * TILE_K_PADDED + k_step], TILE_K_PADDED);
            wmma::load_matrix_sync(B_frag, &current_smem_B[k_step * TILE_N_PADDED + smem_B_n_offset], TILE_N_PADDED);
            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }
        __syncthreads(); 
        current_buffer_idx = next_buffer_idx;
    }

    // --- Epilogue: Store results, add bias, apply ReLU ---
    int smem_C_m_offset = warp_row * WARP_TILE_M;
    int smem_C_n_offset = warp_col * WARP_TILE_N;
    wmma::store_matrix_sync(&smem_C[smem_C_m_offset * TILE_N_PADDED_C + smem_C_n_offset], C_frag, TILE_N_PADDED_C, wmma::mem_row_major);
    __syncthreads(); 

    // Load bias into shared memory
    for (int i = threadIdx.x; i < TILE_N; i += BLOCK_THREADS) {
        const int gmem_n = gmem_n_base + i;
        smem_bias[i] = (gmem_n < N) ? B[gmem_n] : 0.0f;
    }
    __syncthreads();

    // Vectorized write to global memory
    constexpr int VEC_SIZE = 4;
    constexpr int WRITES_PER_THREAD = (TILE_M * TILE_N / VEC_SIZE) / BLOCK_THREADS;

    #pragma unroll
    for (int i = 0; i < WRITES_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        const int m_local = idx / (TILE_N / VEC_SIZE);
        const int n_local_base = (idx % (TILE_N / VEC_SIZE)) * VEC_SIZE;
        
        const int gmem_m = gmem_m_base + m_local;
        const int gmem_n = gmem_n_base + n_local_base;

        if (gmem_m < M && gmem_n < N) {
            float4 val = *reinterpret_cast<float4*>(&smem_C[m_local * TILE_N_PADDED_C + n_local_base]);
            float4 bias_val = *reinterpret_cast<float4*>(&smem_bias[n_local_base]);
            
            val.x += bias_val.x;
            val.y += bias_val.y;
            val.z += bias_val.z;
            val.w += bias_val.w;
            
            val.x = fmaxf(0.0f, val.x);
            val.y = fmaxf(0.0f, val.y);
            val.z = fmaxf(0.0f, val.z);
            val.w = fmaxf(0.0f, val.w);
            
            *reinterpret_cast<float4*>(&Y[gmem_m * output_row_stride + gmem_n]) = val;
        }
    }
}


__global__ void __launch_bounds__(BLOCK_THREADS)
final_linear_kernel_vectorized_padded(
    const float* __restrict__ X,
    const float* __restrict__ W_T,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int M, int N, int K,
    int input_row_stride,
    int output_row_stride
) {
    // This kernel is identical to the fused one but without the ReLU.
    extern __shared__ char smem[];
    half* smem_A = reinterpret_cast<half*>(smem);
    half* smem_B = reinterpret_cast<half*>(smem + SMEM_A_BYTES);
    float* smem_C = reinterpret_cast<float*>(smem + SMEM_A_BYTES + SMEM_B_BYTES);
    float* smem_bias = reinterpret_cast<float*>(smem + SMEM_A_BYTES + SMEM_B_BYTES + SMEM_C_BYTES);

    const int warp_id = threadIdx.x / 32;
    const int block_m_idx = blockIdx.y;
    const int block_n_idx = blockIdx.x;
    const int warp_row = warp_id / WARPS_PER_BLOCK_N;
    const int warp_col = warp_id % WARPS_PER_BLOCK_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
    wmma::fill_fragment(C_frag, 0.0f);

    int current_buffer_idx = 0;
    const int gmem_m_base = block_m_idx * TILE_M;
    const int gmem_n_base = block_n_idx * TILE_N;
    
    load_gmem_tile_to_smem_A_vectorized(X, smem_A, M, K, gmem_m_base, 0, input_row_stride);
    load_gmem_tile_to_smem_B_vectorized(W_T, smem_B, N, K, gmem_n_base, 0, N);
    __syncthreads();

    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += TILE_K) {
        const int next_k_tile_start = k_tile_start + TILE_K;
        const int next_buffer_idx = 1 - current_buffer_idx;
        if (next_k_tile_start < K) {
            half* next_smem_A = smem_A + (next_buffer_idx * TILE_M * TILE_K_PADDED);
            load_gmem_tile_to_smem_A_vectorized(X, next_smem_A, M, K, gmem_m_base, next_k_tile_start, input_row_stride);
            half* next_smem_B = smem_B + (next_buffer_idx * TILE_K * TILE_N_PADDED);
            load_gmem_tile_to_smem_B_vectorized(W_T, next_smem_B, N, K, gmem_n_base, next_k_tile_start, N);
        }
        const half* current_smem_A = smem_A + (current_buffer_idx * TILE_M * TILE_K_PADDED);
        const half* current_smem_B = smem_B + (current_buffer_idx * TILE_K * TILE_N_PADDED);
        #pragma unroll
        for (int k_step = 0; k_step < TILE_K; k_step += WMMA_K) {
            int smem_A_m_offset = warp_row * WARP_TILE_M;
            int smem_B_n_offset = warp_col * WARP_TILE_N;
            wmma::load_matrix_sync(A_frag, &current_smem_A[smem_A_m_offset * TILE_K_PADDED + k_step], TILE_K_PADDED);
            wmma::load_matrix_sync(B_frag, &current_smem_B[k_step * TILE_N_PADDED + smem_B_n_offset], TILE_N_PADDED);
            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }
        __syncthreads(); 
        current_buffer_idx = next_buffer_idx;
    }

    int smem_C_m_offset = warp_row * WARP_TILE_M;
    int smem_C_n_offset = warp_col * WARP_TILE_N;
    wmma::store_matrix_sync(&smem_C[smem_C_m_offset * TILE_N_PADDED_C + smem_C_n_offset], C_frag, TILE_N_PADDED_C, wmma::mem_row_major);
    __syncthreads(); 

    for (int i = threadIdx.x; i < TILE_N; i += BLOCK_THREADS) {
        const int gmem_n = gmem_n_base + i;
        smem_bias[i] = (gmem_n < N) ? B[gmem_n] : 0.0f;
    }
    __syncthreads();

    constexpr int VEC_SIZE = 4;
    constexpr int WRITES_PER_THREAD = (TILE_M * TILE_N / VEC_SIZE) / BLOCK_THREADS;
    #pragma unroll
    for (int i = 0; i < WRITES_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        const int m_local = idx / (TILE_N / VEC_SIZE);
        const int n_local_base = (idx % (TILE_N / VEC_SIZE)) * VEC_SIZE;
        const int gmem_m = gmem_m_base + m_local;
        const int gmem_n = gmem_n_base + n_local_base;

        if (gmem_m < M && gmem_n < N) {
            float4 val = *reinterpret_cast<float4*>(&smem_C[m_local * TILE_N_PADDED_C + n_local_base]);
            float4 bias_val = *reinterpret_cast<float4*>(&smem_bias[n_local_base]);
            val.x += bias_val.x;
            val.y += bias_val.y;
            val.z += bias_val.z;
            val.w += bias_val.w;
            *reinterpret_cast<float4*>(&Y[gmem_m * output_row_stride + gmem_n]) = val;
        }
    }
}


at::Tensor module_fn(
    at::Tensor input,
    const std::vector<at::Tensor>& transposed_weights,
    const std::vector<at::Tensor>& biases
) {
    int num_layers = transposed_weights.size();
    int batch_size = input.size(0);
    auto device = input.device();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    int max_hidden_size = input.size(1);
    for (const auto& w_t : transposed_weights) {
        if (w_t.size(1) > max_hidden_size) max_hidden_size = w_t.size(1);
    }
    
    at::Tensor buffer0 = torch::empty({batch_size, max_hidden_size}, options);
    at::Tensor buffer1 = torch::empty({batch_size, max_hidden_size}, options);

    at::Tensor current_input = input;

    size_t shared_mem_size = SMEM_A_BYTES + SMEM_B_BYTES + SMEM_C_BYTES + SMEM_BIAS_BYTES;
    cudaError_t err = cudaFuncSetAttribute(fused_linear_relu_kernel_vectorized_padded, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    TORCH_CHECK(err == cudaSuccess, "Failed to set dynamic shared memory size for relu kernel");
    err = cudaFuncSetAttribute(final_linear_kernel_vectorized_padded, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    TORCH_CHECK(err == cudaSuccess, "Failed to set dynamic shared memory size for final kernel");

    for (int i = 0; i < num_layers; i++) {
        const at::Tensor& weight_t = transposed_weights[i];
        const at::Tensor& bias = biases[i];
        
        const int M = current_input.size(0);
        const int K = weight_t.size(0);
        const int N = weight_t.size(1);
        
        const int input_row_stride = current_input.stride(0);

        at::Tensor current_output;
        if (i < num_layers - 1) {
            current_output = (i % 2 == 0) ? buffer0.slice(1, 0, N) : buffer1.slice(1, 0, N);
        } else {
            // The final output tensor might not be the max size, so create it here.
            current_output = torch::empty({batch_size, N}, options);
        }
        const int output_row_stride = current_output.stride(0);
        
        dim3 grid_dim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        dim3 block_dim(BLOCK_THREADS);

        if (i < num_layers - 1) { // Apply ReLU for all but the last layer
            fused_linear_relu_kernel_vectorized_padded<<<grid_dim, block_dim, shared_mem_size>>>(
                current_input.data_ptr<float>(), weight_t.data_ptr<float>(), bias.data_ptr<float>(),
                current_output.data_ptr<float>(), M, N, K, input_row_stride, output_row_stride
            );
        } else { // No ReLU for the final layer
            final_linear_kernel_vectorized_padded<<<grid_dim, block_dim, shared_mem_size>>>(
                current_input.data_ptr<float>(), weight_t.data_ptr<float>(), bias.data_ptr<float>(),
                current_output.data_ptr<float>(), M, N, K, input_row_stride, output_row_stride
            );
        }
        current_input = current_output;
    }

    cudaError_t last_err = cudaGetLastError();
    TORCH_CHECK(last_err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(last_err));
    
    return current_input;
}
"""

    def cpp_src_code(self):
        return r"""
#include <torch/extension.h>
#include <vector>

at::Tensor module_fn(
    at::Tensor input,
    const std::vector<at::Tensor>& transposed_weights,
    const std::vector<at::Tensor>& biases
);
"""

    def cuda_cflags(self):
        return [
            "-O3",
            "--use_fast_math",
            "-arch=sm_80",
            "-Xcompiler",
            "-Wno-float-conversion",
            "-std=c++17",
        ]


def gen_inline_cuda():
    src = Solution()
    cuda_module: ModuleType = load_inline(
        name=src.name(),
        cpp_sources=src.cpp_src_code(),
        cuda_sources=src.cuda_src_code(),
        functions=["module_fn"],
        extra_cuda_cflags=src.cuda_cflags(),
        verbose=False,
    )
    return cuda_module


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        self.original_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.padding_multiple = 32  # Required for float4 alignment and tile sizes

        self.padded_sizes = [self._pad_dim(s) for s in self.original_sizes]

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(len(self.original_sizes) - 1):
            in_dim = self.original_sizes[i]
            out_dim = self.original_sizes[i + 1]

            w = nn.Parameter(torch.empty(out_dim, in_dim))
            b = nn.Parameter(torch.empty(out_dim))

            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(b, -bound, bound)

            self.weights.append(w)
            self.biases.append(b)

        self.cuda_module = gen_inline_cuda()
        self._build_padded_buffers()

    def _pad_dim(self, dim):
        return (dim + self.padding_multiple - 1) & ~(self.padding_multiple - 1)

    def _build_padded_buffers(self):
        # Pre-transpose and pad weights/biases into persistent buffers for efficiency.
        # This is done once and buffers are moved to device automatically with .to()
        for i, p in enumerate(self.weights):
            in_dim_orig = self.original_sizes[i]
            out_dim_orig = self.original_sizes[i + 1]
            in_dim_padded = self.padded_sizes[i]
            out_dim_padded = self.padded_sizes[i + 1]

            w_t_padded = torch.zeros(
                in_dim_padded, out_dim_padded, dtype=p.dtype, device=p.device
            )
            with torch.no_grad():
                w_t_padded[:in_dim_orig, :out_dim_orig].copy_(p.T)
            self.register_buffer(f"weight_t_padded_{i}", w_t_padded.contiguous())

        for i, p in enumerate(self.biases):
            out_dim_orig = self.original_sizes[i + 1]
            out_dim_padded = self.padded_sizes[i + 1]

            b_padded = torch.zeros(out_dim_padded, dtype=p.dtype, device=p.device)
            with torch.no_grad():
                b_padded[:out_dim_orig].copy_(p)
            self.register_buffer(f"bias_padded_{i}", b_padded.contiguous())

    def to(self, *args, **kwargs):
        # Ensure buffers are rebuilt if the device changes
        super().to(*args, **kwargs)
        self._build_padded_buffers()
        return self

    def forward(self, x):
        in_dim_orig = self.original_sizes[0]
        in_dim_padded = self.padded_sizes[0]

        # Pad input tensor if necessary
        if x.shape[1] != in_dim_padded:
            # This path is taken if input is not already padded
            x_padded = torch.nn.functional.pad(x, (0, in_dim_padded - in_dim_orig))
        else:
            x_padded = x

        transposed_weights = [
            getattr(self, f"weight_t_padded_{i}") for i in range(len(self.weights))
        ]
        biases = [getattr(self, f"bias_padded_{i}") for i in range(len(self.biases))]

        output_padded = self.cuda_module.module_fn(x_padded, transposed_weights, biases)

        # Unpad the final output tensor to match the original expected output size
        out_dim_orig = self.original_sizes[-1]
        return output_padded.narrow(1, 0, out_dim_orig)
