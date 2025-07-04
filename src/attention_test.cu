#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "attention.hu"
#include "tensor.hu"

int main() {
    const int seq_len = 2;
    const int d_k = 3;

    float h_Q[2][3] = {{1, 0, 0}, {0, 1, 0}};
    float h_K[2][3] = {{1, 2, 3}, {4, 5, 6}};
    float h_V[2][3] = {{7, 8, 9}, {10, 11, 12}};

    int shape[2] = {seq_len, d_k};
    Tensor* Q = tensor_new(2, shape);
    memcpy(Q->data, h_Q, seq_len * d_k * sizeof(float));

    Tensor* K = tensor_new(2, shape);
    memcpy(K->data, h_K, seq_len * d_k * sizeof(float));

    Tensor* V = tensor_new(2, shape);
    memcpy(V->data, h_V, seq_len * d_k * sizeof(float));

    // tensor_show(Q);
    // tensor_show(K);
    // tensor_show(V);

    Tensor* attn_out = tensor_new(2, shape);

    scaled_dot_product_attention(Q, K, V, attn_out, seq_len, d_k);

    tensor_show(attn_out);

    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    tensor_free(attn_out);
    cudaDeviceReset();
    return 0;
}
