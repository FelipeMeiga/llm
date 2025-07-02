#include <math.h>
#include "attention.hu"

__global__ void softmax_row(float* mat, int cols) {
    int row = blockIdx.x;
    float* ptr = mat + row * cols;
    float maxv = -INFINITY;
    
    for (int j = 0; j < cols; j++) {
        maxv = fmaxf(maxv, ptr[j]);
    }
    
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        ptr[j] = expf(ptr[j] - maxv);
        sum += ptr[j];
    }
    
    for (int j = 0; j < cols; j++) {
        ptr[j] /= sum;
    }
}

void scaled_dot_product_attention(Tensor* Q, Tensor* K, Tensor* V, Tensor* attn_out, int seq_len, int d_k) {
    Tensor* Kt = tensor_new(2, (int[]){ d_k, seq_len });
    tensor_transpose_cuda(K, Kt, 0, 1);

    Tensor* scores = tensor_new(2, (int[]){ seq_len, seq_len });
    tensor_matmul_cuda(scores, Q, Kt);

    float scale = 1.0f / sqrtf((float)d_k);
    tensor_scale_cuda(scores, scores, scale);

    softmax_row<<<seq_len, 1>>>(scores->data, seq_len);
    cudaDeviceSynchronize();

    tensor_matmul_cuda(attn_out, scores, V);

    tensor_free(Kt);
    tensor_free(scores);
}
