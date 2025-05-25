#include <stdio.h>
#include <stdlib.h>

#include "nn.h"
#include "tensor.h"

Linear* linear_new(int in_dim, int out_dim) {
    Linear *lin = malloc(sizeof(Linear));
    if (!lin) {
        fprintf(stderr, "linear_new: failed to allocate Linear struct\n");
        exit(EXIT_FAILURE);
    }

    lin->in_dim  = in_dim;
    lin->out_dim = out_dim;

    int shapeW[2] = { in_dim, out_dim };
    lin->W = tensor_rand(2, shapeW);
    if (!lin->W) {
        fprintf(stderr, "linear_new: failed to allocate weight tensor W\n");
        exit(EXIT_FAILURE);
    }

    int shapeB[1] = { out_dim };
    lin->b = tensor_zeros(1, shapeB);
    if (!lin->b) {
        fprintf(stderr, "linear_new: failed to allocate bias tensor b\n");
        exit(EXIT_FAILURE);
    }

    lin->dW = tensor_zeros(2, shapeW);
    if (!lin->dW) {
        fprintf(stderr, "linear_new: failed to allocate gradient tensor dW\n");
        exit(EXIT_FAILURE);
    }

    lin->db = tensor_zeros(1, shapeB);
    if (!lin->db) {
        fprintf(stderr, "linear_new: failed to allocate gradient tensor db\n");
        exit(EXIT_FAILURE);
    }

    return lin;
}

void linear_show(Linear* l) {
    printf("=================================================Header Linear:\n");
    printf("in_dim: %d\nout_dim: %d\n", l->in_dim, l->out_dim);
    printf("=================================================");
    printf("Weights tensor: \n");
    tensor_show(l->W);
    printf("=================================================");
    printf("Bias tensor: \n");
    tensor_show(l->b);
    printf("=================================================");
    printf("Weights gradients tensor: \n");
    tensor_show(l->dW);
    printf("=================================================");
    printf("Bias gradients tensor: \n");
    tensor_show(l->db);
    printf("=================================================\n");
}

void linear_free(Linear *lin) {
    tensor_free(lin->b);
    tensor_free(lin->db);
    tensor_free(lin->dW);
    tensor_free(lin->W);
    free(lin);
}

Tensor* linear_forward(Linear *lin, const Tensor *X) {
    if (!lin || !X) {
        fprintf(stderr, "linear_forward: NULL pointer argument\n");
        exit(EXIT_FAILURE);
    }
    if (X->ndim != 2 || X->shape[1] != lin->in_dim) {
        fprintf(stderr,
                "linear_forward: input X must be 2D with shape[1]=in_dim (got ndim=%d, shape[1]=%d, in_dim=%d)\n",
                X->ndim, X->shape[1], lin->in_dim);
        exit(EXIT_FAILURE);
    }

    Tensor *Y = tensor_matmul(X, lin->W);
    if (!Y) {
        fprintf(stderr, "linear_forward: failed to compute X·W\n");
        exit(EXIT_FAILURE);
    }

    int batch    = X->shape[0];
    int out_dim  = lin->out_dim;
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            size_t idxY = i * Y->stride[0] + j * Y->stride[1];
            Y->data[idxY] += lin->b->data[j];
        }
    }

    return Y;
}
