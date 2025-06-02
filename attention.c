#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "attention.h"

Tensor *scaled_dot_product_attention(const Tensor *Q, const Tensor *K, const Tensor *V) {
    if (Q->ndim != 2 || K->ndim != 2 || V->ndim != 2) {
        fprintf(stderr, "scaled_dot_product_attention: Q, K e V must be 2D\n");
        exit(EXIT_FAILURE);
    }
    int Tq = Q->shape[0], Dq = Q->shape[1];
    int Tk = K->shape[0], Dk = K->shape[1];
    int Tv = V->shape[0], Dv = V->shape[1];
    if (Dq != Dk || Dk != Dv) {
        fprintf(stderr, "scaled_dot_product_attention: inconsistent dimensions D (Q:%d, K:%d, V:%d)\n", Dq, Dk, Dv);
        exit(EXIT_FAILURE);
    }
    int T = Tq;
    if (Tk != T || Tv != T) {
        fprintf(stderr, "scaled_dot_product_attention: inconsistent dimensions T (Q:%d, K:%d, V:%d)\n", Tq, Tk, Tv);
        exit(EXIT_FAILURE);
    }
    int D = Dq;

    Tensor *Kt = tensor_transpose(K, 0, 1);
    Tensor *scores = tensor_matmul(Q, Kt);
    tensor_free(Kt);

    float scale = 1.0f / sqrtf((float)D);

    Tensor *scaled_scores = tensor_new(scores->ndim, scores->shape);
    if (!scaled_scores) {
        fprintf(stderr, "scaled_dot_product_attention: failed to allocate scaled_scores\n");
        exit(EXIT_FAILURE);
    }
    tensor_scale(scaled_scores, scores, scale);
    tensor_free(scores);

    tensor_softmax(scaled_scores, 1);

    Tensor *C = tensor_matmul(scaled_scores, V);

    tensor_free(scaled_scores);

    return C;
}