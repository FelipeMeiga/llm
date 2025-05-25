#include <stdio.h>

#include "nn.h"
#include "tensor.h"


int main(void) {
    int batch = 2;  // quantos exemplos por vez
    int in_dim = 3;  // tamanho do vetor de entrada
    int hidden_dim = 4;  // tamanho da camada oculta
    int out_dim = 2;  // tamanho da saída

    Linear *l1 = linear_new(in_dim, hidden_dim);
    Linear *l2 = linear_new(hidden_dim, out_dim);

    int shapeX[2] = { batch, in_dim };
    Tensor *X = tensor_rand(2, shapeX);

    Tensor *Y1 = linear_forward(l1, X);
    Tensor *Y2 = linear_forward(l2, Y1);

    printf("=== Input X (%dx%d) ===\n", batch, in_dim);
    tensor_show(X);

    printf("\n=== After Layer 1 (%dx%d) ===\n", batch, hidden_dim);
    tensor_show(Y1);

    printf("\n=== After Layer 2 (%dx%d) ===\n", batch, out_dim);
    tensor_show(Y2);

    tensor_free(X);
    tensor_free(Y1);
    tensor_free(Y2);

    linear_free(l1);
    linear_free(l2);

    return 0;
}