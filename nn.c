#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    lin->X = NULL;

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
    if (X->ndim!=2 || X->shape[1]!=lin->in_dim) {
        fprintf(stderr, "linear_forward: expected X shape [B,%d], got [%d,%d]\n", lin->in_dim, X->shape[0], X->shape[1]);
        exit(EXIT_FAILURE);
    }

    lin->X = (Tensor*)X;

    Tensor *Y = tensor_matmul(X, lin->W);
    int B = X->shape[0];
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < lin->out_dim; ++j) {
            size_t idx = i*Y->stride[0] + j*Y->stride[1];
            Y->data[idx] += lin->b->data[j];
        }
    }
    return Y;
}

Tensor* linear_backward(Linear *lin, const Tensor *dY) {
    if (!lin->X) {
        fprintf(stderr, "linear_backward: no saved input (lin->X is NULL)\n");
        exit(EXIT_FAILURE);
    }
    Tensor *X = lin->X;
    int B   = X->shape[0];
    int Din = lin->in_dim;
    int Dout= lin->out_dim;

    Tensor *Xt = tensor_transpose(X, 0, 1);
    Tensor *new_dW = tensor_matmul(Xt, dY);
    tensor_free(Xt);
    tensor_free(lin->dW);
    lin->dW = new_dW;

    for (int j = 0; j < Dout; ++j) {
        float s = 0.0f;
        for (int i = 0; i < B; ++i) {
            size_t idx = i*dY->stride[0] + j*dY->stride[1];
            s += dY->data[idx];
        }
        lin->db->data[j] = s;
    }

    Tensor *Wt = tensor_transpose(lin->W, 0, 1);
    Tensor *dX = tensor_matmul(dY, Wt);
    tensor_free(Wt);

    return dX;
}
