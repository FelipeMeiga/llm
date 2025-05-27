#ifndef NN_H
#define NN_H

#include "tensor.h"

typedef struct {
    int in_dim;
    int out_dim;
    Tensor *W;
    Tensor *b;
    Tensor *dW;
    Tensor *db;
    Tensor *X;
} Linear;

Linear* linear_new(int in_dim, int out_dim);
void linear_show(Linear* l);
void linear_free(Linear *lin);

Tensor* linear_forward(Linear *lin, const Tensor *X);
Tensor* linear_backward(Linear *lin, const Tensor *dY);

#endif
