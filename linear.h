#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

typedef struct {
    int in_dim, out_dim;
    Tensor *W, *b;      // parâmetros
    Tensor *dW, *db;    // gradientes
    
    Tensor *mW, *vW;
    Tensor *mb, *vb;
    int t;
    Tensor *X;
} Linear;

Linear* linear_new(int in_dim, int out_dim);
void linear_show(Linear* l);
void linear_free(Linear *lin);

Tensor* linear_forward(Linear *lin, const Tensor *X);
Tensor* linear_backward(Linear *lin, const Tensor *dY);
void linear_update_adam(Linear *lin, float lr, float beta1, float beta2, float eps);

#endif
