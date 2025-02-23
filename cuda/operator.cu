// masked_softmax.cu
#include <cuda_fp16.h>
#include <cfloat>

extern "C" __global__ void masked_softmax_kernel(
    float* data,
    const int batch_size,
    const int seq_len,
    const int total_seq_len
) {
    // 计算当前处理的行索引
    const int row_idx = blockIdx.x;
    const int b = row_idx / seq_len;
    const int i = row_idx % seq_len;
    
    // 计算有效元素边界
    const int boundary = total_seq_len - seq_len + i + 1;
    const int offset = b * seq_len * total_seq_len + i * total_seq_len;

    extern __shared__ float shared_mem[];
    float* max_shared = shared_mem;
    float* sum_shared = &shared_mem[blockDim.x];

    // 阶段1: 计算最大值
    float thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j < boundary; j += blockDim.x) {
        thread_max = fmaxf(thread_max, data[offset + j]);
    }
    max_shared[threadIdx.x] = thread_max;
    __syncthreads();

    // 归约最大值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            max_shared[threadIdx.x] = fmaxf(max_shared[threadIdx.x], max_shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = max_shared[0];
    __syncthreads();

    // 阶段2: 计算指数和
    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j < boundary; j += blockDim.x) {
        const float val = data[offset + j];
        const float exp_val = expf(val - row_max);
        data[offset + j] = exp_val;
        thread_sum += exp_val;
    }
    sum_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    // 归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_shared[threadIdx.x] += sum_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float row_sum = sum_shared[0];
    __syncthreads();

    // 阶段3: 归一化并置零
    for (int j = threadIdx.x; j < total_seq_len; j += blockDim.x) {
        if (j < boundary) {
            data[offset + j] /= row_sum;
        } else {
            data[offset + j] = 0.0f;
        }
    }
}
extern "C" __global__ void matmul_transb_kernel(
    const float* A, const float* B, float* C,
    int dim1, int dim2, int dim3,
    float alpha, float beta)
{
    // 获取当前线程的全局索引
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引

    // 检查是否超出矩阵范围
    if (row >= dim1 || col >= dim2) return;

    // 计算矩阵乘法结果
    float sum = 0.0f;
    for (int k = 0; k < dim3; ++k) {
        sum += A[row * dim3 + k] * B[col * dim3 + k]; // 避免显式转置
    }

    // 更新矩阵 C 的值
    int index = row * dim2 + col;
    C[index] = alpha * sum + beta * C[index];
}
extern "C" __global__ void rms_norm_kernel(
    const float* x, const float* w, float* y,
    int dim, int batch, float epsilon)
{
    // 获取当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查是否超出数组范围
    if (idx >= batch * dim) return;

    // 计算当前样本的起始位置和元素索引
    int sample_idx = idx / dim; // 当前样本的索引
    int elem_idx = idx % dim;   // 当前样本内的元素索引

    // 计算均方根（RMS）
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float x_val = x[sample_idx * dim + i];
        sum_sq += x_val * x_val;
    }
    float rms = sqrtf(sum_sq / dim + epsilon);

    // 更新 y 的值：y = (x * w) / rms
    float x_val = x[idx];
    float w_val = w[elem_idx];
    y[idx] = (x_val * w_val) / rms;
}
extern "C" __global__ void rope_kernel(
    float* y, int seq_len, int n_heads, int d, int start_pos, float theta)
{
    // 获取当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查是否超出数组范围
    if (idx >= seq_len * n_heads * d) return;

    // 计算当前 token、head 和维度索引
    int tok = idx / (n_heads * d);       // 当前 token 索引
    int head = (idx % (n_heads * d)) / d; // 当前 head 索引
    int i = (idx % d) / 2;               // 当前维度索引（每两个元素一组）

    // 只处理一半的维度（复数对）
    if ((idx % d) >= d / 2) return;

    // 计算位置和频率
    float pos = static_cast<float>(start_pos + tok);
    float freq = pos / powf(theta, (2.0f * i) / static_cast<float>(d));
    float sin_val, cos_val;
    sincosf(freq, &sin_val, &cos_val);

    // 获取原始值
    int idx1 = tok * n_heads * d + head * d + i;
    int idx2 = idx1 + d / 2;
    float a = y[idx1];
    float b = y[idx2];

    // 更新旋转后的值
    y[idx1] = a * cos_val - b * sin_val;
    y[idx2] = b * cos_val + a * sin_val;
}
extern "C" __global__ void swiglu_kernel(
    const float* x, float* y, int len)
{
    // 获取当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查是否超出数组范围
    if (idx >= len) return;

    // 计算 SiLU(x) = x * sigmoid(x)
    float sigmoid_x = 1.0f / (1.0f + expf(-x[idx]));
    float silu_x = x[idx] * sigmoid_x;

    // 更新 y 的值：y = silu(x) * y
    y[idx] = silu_x * y[idx];
}