extern "C" __global__ void masked_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int seq_len,
    int total_seq_len,
    int batch
) {
    int b = blockIdx.x; 
    int i = threadIdx.x;

    if (i >= seq_len || b >= batch) return;

    int base = b * seq_len * total_seq_len + i * total_seq_len;
    int boundary = total_seq_len - seq_len + i + 1;


    float max_val = input[base];
    for (int j = 0; j < boundary; j++) {
        max_val = fmaxf(max_val, input[base + j]);
    }


    float sum = 0.0f;
    for (int j = 0; j < boundary; j++) {
        float exp_val = expf(input[base + j] - max_val);
        output[base + j] = exp_val;
        sum += exp_val;
    }


    for (int j = 0; j < boundary; j++) {
        output[base + j] /= sum;
    }

    for (int j = boundary; j < total_seq_len; j++) {
        output[base + j] = 0.0f;
    }
}
extern "C" __global__ void matmul_transb_kernel(
    const float* A, const float* B, float* C,
    int dim1, int dim2, int dim3,
    float alpha, float beta)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 


    if (row >= dim1 || col >= dim2) return;


    float sum = 0.0f;
    for (int k = 0; k < dim3; ++k) {
        sum += A[row * dim3 + k] * B[col * dim3 + k]; 
    }


    int index = row * dim2 + col;
    C[index] = alpha * sum + beta * C[index];
}
extern "C" __global__ void rms_norm_kernel(
    const float* x, const float* w, float* y,
    int dim, int batch, float epsilon)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx >= batch * dim) return;


    int sample_idx = idx / dim;
    int elem_idx = idx % dim;  

    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float x_val = x[sample_idx * dim + i];
        sum_sq += x_val * x_val;
    }
    float rms = sqrtf(sum_sq / dim + epsilon);


    float x_val = x[idx];
    float w_val = w[elem_idx];
    y[idx] = (x_val * w_val) / rms;
}
extern "C" __global__ void rope_kernel(
    float* y, int seq_len, int n_heads, int d, int start_pos, float theta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * n_heads * d) return;
    int tok = idx / (n_heads * d);       
    int head = (idx % (n_heads * d)) / d;
    int i = (idx % d) / 2;              
    if ((idx % d) >= d / 2) return;
    float pos = static_cast<float>(start_pos + tok);
    float freq = pos / powf(theta, (2.0f * i) / static_cast<float>(d));
    float sin_val, cos_val;
    sincosf(freq, &sin_val, &cos_val);
    int idx1 = tok * n_heads * d + head * d + i;
    int idx2 = idx1 + d / 2;
    float a = y[idx1];
    float b = y[idx2];
    y[idx1] = a * cos_val - b * sin_val;
    y[idx2] = b * cos_val + a * sin_val;
}
extern "C" __global__ void swiglu_kernel(
    const float* x, float* y, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    float sigmoid_x = 1.0f / (1.0f + expf(-x[idx]));
    float silu_x = x[idx] * sigmoid_x;
    y[idx] = silu_x * y[idx];
}