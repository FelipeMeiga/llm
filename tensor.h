#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

typedef struct {
    float *data;
    int   ndim;
    int  *shape;
    int  *stride;
    size_t size;
} Tensor;

Tensor* tensor_new(int ndim, const int *shape);
void tensor_free(Tensor *t);

Tensor* tensor_zeros(int ndim, const int *shape);
Tensor* tensor_ones(int ndim, const int *shape);
Tensor* tensor_rand(int ndim, const int *shape);

void tensor_add(Tensor *out, const Tensor *A, const Tensor *B);
void tensor_sub(Tensor *out, const Tensor *A, const Tensor *B);
void tensor_mul(Tensor *out, const Tensor *A, const Tensor *B);
void tensor_scale(Tensor *out, const Tensor *A, float alpha);

Tensor* tensor_matmul(const Tensor *A, const Tensor *B);

Tensor* tensor_reshape(const Tensor *A, int ndim, const int *new_shape);
Tensor* tensor_transpose(const Tensor *A, int dim0, int dim1);

void tensor_relu(Tensor *A);
Tensor* tensor_relu_backward(const Tensor *pre, const Tensor *dA);
void tensor_softmax(Tensor *A, int axis);
Tensor* tensor_sigmoid(Tensor *A);
Tensor* tensor_sigmoid_backward(const Tensor *sig, const Tensor *dA);

float tensor_get(const Tensor *t, const int *coords);
void tensor_set(Tensor *t, const int *coords, float value);

void tensor_show(Tensor *t);

#endif
