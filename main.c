#include <stdio.h>
#include <stdlib.h>

#include "nn.h"
#include "tensor.h"

int main(void) {
    int batch      = 2;
    int in_dim     = 3;
    int hidden_dim = 4;
    int out_dim    = 2;

    Linear *l1 = linear_new(in_dim,     hidden_dim);
    Linear *l2 = linear_new(hidden_dim, out_dim);

    int shapeX[2] = { batch, in_dim };
    Tensor *X = tensor_rand(2, shapeX);
    printf("=== X ===\n");
    tensor_show(X);

    Tensor *Y1 = linear_forward(l1, X);
    printf("\n=== Y1 (pre-ReLU) ===\n");
    tensor_show(Y1);

    Tensor *A1 = tensor_reshape(Y1, Y1->ndim, Y1->shape);
    tensor_relu(Y1);
    printf("\n=== A1 (post-ReLU) ===\n");
    tensor_show(Y1);

    Tensor *Y2 = linear_forward(l2, Y1);
    printf("\n=== Y2 (output) ===\n");
    tensor_show(Y2);

    int shapeY2[2] = { batch, out_dim };
    Tensor *dY2 = tensor_ones(2, shapeY2);
    printf("\n=== dY2 (dL/dY2) ===\n");
    tensor_show(dY2);

    Tensor *dA1 = linear_backward(l2, dY2);
    printf("\n=== dA1 (dL/dA1 pre-ReLU) ===\n");
    tensor_show(dA1);

    Tensor *dY1 = tensor_relu_backward(A1, dA1);
    printf("\n=== dY1 (∂L/∂Y1 pós-ReLU) ===\n");
    tensor_show(dY1);

    Tensor *dX = linear_backward(l1, dY1);
    printf("\n=== dX (∂L/∂X) ===\n");
    tensor_show(dX);

    tensor_free(X);
    tensor_free(Y1);
    tensor_free(A1);
    tensor_free(Y2);
    tensor_free(dY2);
    tensor_free(dA1);
    tensor_free(dY1);
    tensor_free(dX);
    linear_free(l1);
    linear_free(l2);

    return 0;
}
