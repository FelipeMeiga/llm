#include <math.h>
#include "tensor.hu"

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
    int shapeKt[2] = { d_k, seq_len };
    Tensor* Kt = tensor_new(2, shapeKt);
    tensor_transpose_cuda(K, Kt, 0, 1);
    
    int shapeScores[2] = { seq_len, seq_len };
    Tensor* scores = tensor_new(2, shapeScores);
    tensor_matmul_cuda(scores, Q, Kt);

    float scale = 1.0f / sqrtf((float)d_k);
    tensor_scale_cuda(scores, scores, scale);

    float *scores_data_gpu = NULL;
    cudaMalloc((void**)&scores_data_gpu, sizeof(float)*scores->size);
    cudaMemcpy(scores_data_gpu, scores->data, sizeof(float)*scores->size, cudaMemcpyHostToDevice);
    
    softmax_row<<<seq_len, 1>>>(scores_data_gpu, seq_len);
    cudaDeviceSynchronize();

    cudaMemcpy(scores->data, scores_data_gpu, sizeof(float)*scores->size, cudaMemcpyDeviceToHost);
    cudaFree(scores_data_gpu);

    // tensor_show(scores);

    tensor_matmul_cuda(attn_out, scores, V);

    tensor_free(Kt);
    tensor_free(scores);
}
