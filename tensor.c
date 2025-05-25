#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "tensor.h"
#include "utils.h"

Tensor *tensor_new(int ndim, const int *shape) {
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
        fprintf(stderr, "Error allocating Tensor\n");
        return NULL;
    }

    t->ndim = ndim;

    t->shape = (int*)malloc(ndim * sizeof(int));
    if (!t->shape) {
        fprintf(stderr, "Error allocating shape\n");
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int));

    size_t size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    t->size = size;

    t->stride = (int*)malloc(ndim * sizeof(int));
    if (!t->stride) {
        fprintf(stderr, "Error allocating stride\n");
        free(t->shape);
        free(t);
        return NULL;
    }

    t->stride[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--) {
        t->stride[i] = t->stride[i+1] * shape[i+1];
    }

    t->data = (float*)malloc(size * sizeof(float));
    if (!t->data) {
        fprintf(stderr, "Error allocating data\n");
        free(t->stride);
        free(t->shape);
        free(t);
        return NULL;
    }

    memset(t->data, 0, size * sizeof(float));

    return t;
}

static size_t tensor_index(const Tensor *t, const int *coords) {
    size_t offset = 0;
    for (int d = 0; d < t->ndim; ++d) {
        if (coords[d] < 0 || coords[d] >= t->shape[d]) {
            fprintf(stderr,
                    "tensor_index: coord %d out of bounds for dimension %d (shape=%d)\n",
                    coords[d], d, t->shape[d]);
            exit(EXIT_FAILURE);
        }
        offset += coords[d] * t->stride[d];
    }
    return offset;
}

void __tensor_print_recursive(const Tensor *t, int dim, int *coords) {
    if (dim == t->ndim) {
        size_t idx = tensor_index(t, coords);
        printf("%.2f", t->data[idx]);
        return;
    }

    printf("[");
    for (int i = 0; i < t->shape[dim]; ++i) {
        coords[dim] = i;
        __tensor_print_recursive(t, dim + 1, coords);
        if (i < t->shape[dim] - 1) {
            printf(", ");
        }
    }
    printf("]");
}

void tensor_free(Tensor *t) {
    free(t->shape);
    free(t->stride);
    free(t->data);
    free(t);
}

void tensor_show(Tensor *t) {
    printf("ndim: %d\n", t->ndim);
    printf("size: %zu\n", t->size);

    printf("shape: ");
    __array_print(t->shape, t->ndim, sizeof(int), __int_print);

    printf("stride: ");
    __array_print(t->stride, t->ndim, sizeof(int), __int_print);

    printf("data:\n");
    int *coords = malloc(t->ndim * sizeof(int));
    if (!coords) {
        fprintf(stderr, "Error allocating coords\n");
        return;
    }
    __tensor_print_recursive(t, 0, coords);
    printf("\n");
    free(coords);
}

Tensor* tensor_zeros(int ndim, const int *shape) {
    Tensor *t = tensor_new(ndim, shape);
    if (!t) return NULL;

    for (size_t i = 0; i < t->size; ++i) {
        t->data[i] = 0.0f;
    }

    return t;
}

Tensor* tensor_ones(int ndim, const int *shape) {
    Tensor *t = tensor_new(ndim, shape);
    if (!t) return NULL;

    for (size_t i = 0; i < t->size; ++i) {
        t->data[i] = 1.0f;
    }

    return t;
}

Tensor* tensor_rand(int ndim, const int *shape) {
    Tensor *t = tensor_new(ndim, shape);
    if (!t) return NULL;

    srand((unsigned)time(NULL));

    for (size_t i = 0; i < t->size; ++i) {
        t->data[i] = (float)rand() / (float)RAND_MAX;
    }

    return t;
}

void tensor_add(Tensor *out, const Tensor *A, const Tensor *B) {
    if (A->ndim != B->ndim || A->ndim != out->ndim) {
        fprintf(stderr, "tensor_add: incompatible dimensions\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != out->shape[i]) {
            fprintf(stderr, "tensor_add: diferent shapes in dimension %d (A=%d, B=%d, out=%d)\n", i, A->shape[i], B->shape[i], out->shape[i]);
            exit(EXIT_FAILURE);
        }
    }

    size_t n = A->size;
    for (size_t i = 0; i < n; ++i) {
        out->data[i] = A->data[i] + B->data[i];
    }
}

void tensor_sub(Tensor *out, const Tensor *A, const Tensor *B) {
    if (A->ndim != B->ndim || A->ndim != out->ndim) {
        fprintf(stderr, "tensor_add: incompatible dimensions\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != out->shape[i]) {
            fprintf(stderr, "tensor_add: diferent shapes in dimension %d (A=%d, B=%d, out=%d)\n", i, A->shape[i], B->shape[i], out->shape[i]);
            exit(EXIT_FAILURE);
        }
    }

    size_t n = A->size;
    for (size_t i = 0; i < n; ++i) {
        out->data[i] = A->data[i] - B->data[i];
    }
}

void tensor_scale(Tensor *out, const Tensor *A, float alpha) {
    if (A->ndim != out->ndim) {
        fprintf(stderr, "tensor_scale: incompatible dimensions (A.ndim=%d, out.ndim=%d)\n",
                A->ndim, out->ndim);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != out->shape[i]) {
            fprintf(stderr, "tensor_scale: diferent shapes in dimension %d (A=%d, out=%d)\n", i, A->shape[i], out->shape[i]);
            exit(EXIT_FAILURE);
        }
    }

    size_t n = A->size;
    for (size_t i = 0; i < n; ++i) {
        out->data[i] = A->data[i] * alpha;
    }
}

void tensor_mul(Tensor *out, const Tensor *A, const Tensor *B) {
    if (A->ndim != B->ndim || A->ndim != out->ndim) {
        fprintf(stderr,
                "tensor_mul: dimension mismatch (A.ndim=%d, B.ndim=%d, out.ndim=%d)\n", A->ndim, B->ndim, out->ndim);
        exit(EXIT_FAILURE);
    }
    for (int d = 0; d < A->ndim; ++d) {
        if (A->shape[d] != B->shape[d] || A->shape[d] != out->shape[d]) {
            fprintf(stderr, "tensor_mul: shape mismatch at dim %d (A=%d, B=%d, out=%d)\n", d, A->shape[d], B->shape[d], out->shape[d]);
            exit(EXIT_FAILURE);
        }
    }

    size_t n = A->size;
    for (size_t i = 0; i < n; ++i) {
        out->data[i] = A->data[i] * B->data[i];
    }
}

Tensor* tensor_matmul(const Tensor *A, const Tensor *B) {
    if (A->ndim != 2 || B->ndim != 2) {
        fprintf(stderr, "tensor_matmul: only 2d tensors are suported (A.ndim=%d, B.ndim=%d)\n", A->ndim, B->ndim);
        exit(EXIT_FAILURE);
    }
    int m = A->shape[0];
    int n = A->shape[1];
    int n2 = B->shape[0];
    int p = B->shape[1];
    if (n != n2) {
        fprintf(stderr, "tensor_matmul: incompatibility with inner dimension (A.cols=%d, B.rows=%d)\n", n, n2);
        exit(EXIT_FAILURE);
    }

    int out_shape[2] = { m, p };
    Tensor *C = tensor_new(2, out_shape);
    if (!C) {
        fprintf(stderr, "tensor_matmul: failed to allocate output tensor\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                size_t idxA = i * A->stride[0] + k * A->stride[1];
                size_t idxB = k * B->stride[0] + j * B->stride[1];
                sum += A->data[idxA] * B->data[idxB];
            }
            size_t idxC = i * C->stride[0] + j * C->stride[1];
            C->data[idxC] = sum;
        }
    }

    return C;
}

Tensor* tensor_reshape(const Tensor *A, int ndim, const int *new_shape) {
    
    size_t new_size = 1;
    for (int i = 0; i < ndim; ++i) {
        new_size *= new_shape[i];
    }
    
    if (new_size != A->size) {
        fprintf(stderr,
                "tensor_reshape: invalid dims product (wants %zu elements, but tensor has %zu)\n",
                new_size, A->size);
        exit(EXIT_FAILURE);
    }
    
    Tensor *t = tensor_new(ndim, new_shape);
    if (!t) {
        fprintf(stderr, "tensor_reshape: failed to allocate new tensor\n");
        exit(EXIT_FAILURE);
    }
    
    memcpy(t->data, A->data, A->size * sizeof(float));
    return t;
}

Tensor* tensor_transpose(const Tensor *A, int dim0, int dim1) {
    if (A->ndim < 2) {
        fprintf(stderr, "tensor_transpose: tensor must have >= 2 dims (ndim=%d)\n", A->ndim);
        exit(EXIT_FAILURE);
    }
    if (dim0 < 0 || dim0 >= A->ndim || dim1 < 0 || dim1 >= A->ndim) {
        fprintf(stderr, "tensor_transpose: invalid indexes (dim0=%d, dim1=%d)\n", dim0, dim1);
        exit(EXIT_FAILURE);
    }

    int *new_shape = malloc(A->ndim * sizeof(int));
    if (!new_shape) {
        fprintf(stderr, "tensor_transpose: failed to allocate new_shape\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->ndim; ++i) {
        new_shape[i] = A->shape[i];
    }
    
    new_shape[dim0] = A->shape[dim1];
    new_shape[dim1] = A->shape[dim0];

    Tensor *T = tensor_new(A->ndim, new_shape);
    free(new_shape);
    if (!T) {
        fprintf(stderr, "tensor_transpose: failed to create output tensor\n");
        exit(EXIT_FAILURE);
    }

    int *coords     = malloc(A->ndim * sizeof(int));
    int *new_coords = malloc(A->ndim * sizeof(int));
    if (!coords || !new_coords) {
        fprintf(stderr, "tensor_transpose: failed to allocate coords\n");
        exit(EXIT_FAILURE);
    }

    for (size_t idx = 0; idx < A->size; ++idx) {
        size_t rem = idx;
        for (int d = 0; d < A->ndim; ++d) {
            coords[d] = rem / A->stride[d];
            rem       = rem % A->stride[d];
        }

        for (int d = 0; d < A->ndim; ++d) {
            new_coords[d] = coords[d];
        }
        new_coords[dim0] = coords[dim1];
        new_coords[dim1] = coords[dim0];

        size_t new_offset = 0;
        for (int d = 0; d < A->ndim; ++d) {
            new_offset += new_coords[d] * T->stride[d];
        }
        T->data[new_offset] = A->data[idx];
    }

    free(coords);
    free(new_coords);
    return T;
}

void tensor_relu(Tensor *A) {
    if (!A) {
        fprintf(stderr, "tensor_relu: input tensor is NULL\n");
        exit(EXIT_FAILURE);
    }
    size_t n = A->size;
    for (size_t i = 0; i < n; ++i) {
        if (A->data[i] < 0.0f) {
            A->data[i] = 0.0f;
        }
    }
}

static void _vector_softmax(Tensor *A, int axis, int *coords) {
    int len = A->shape[axis];
    size_t stride = A->stride[axis];

    coords[axis] = 0;
    size_t base = 0;
    for (int d = 0; d < A->ndim; ++d) {
        base += coords[d] * A->stride[d];
    }

    float maxv = -INFINITY;
    for (int i = 0; i < len; ++i) {
        float v = A->data[base + i * stride];
        if (v > maxv) maxv = v;
    }

    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        float e = expf(A->data[base + i * stride] - maxv);
        A->data[base + i * stride] = e;
        sum += e;
    }

    for (int i = 0; i < len; ++i) {
        A->data[base + i * stride] /= sum;
    }
}

static void _softmax_rec(Tensor *A, int axis, int dim, int *coords) {
    if (dim == A->ndim) {
        _vector_softmax(A, axis, coords);
        return;
    }
    if (dim == axis) {
        coords[dim] = 0;
        _softmax_rec(A, axis, dim + 1, coords);
    } else {
        for (int i = 0; i < A->shape[dim]; ++i) {
            coords[dim] = i;
            _softmax_rec(A, axis, dim + 1, coords);
        }
    }
}

void tensor_softmax(Tensor *A, int axis) {
    if (!A) {
        fprintf(stderr, "tensor_softmax: input tensor is NULL\n");
        exit(EXIT_FAILURE);
    }
    if (axis < 0 || axis >= A->ndim) {
        fprintf(stderr,
                "tensor_softmax: axis %d is out of bounds for tensor with ndim=%d\n",
                axis, A->ndim);
        exit(EXIT_FAILURE);
    }

    int *coords = malloc(A->ndim * sizeof(int));
    if (!coords) {
        fprintf(stderr, "tensor_softmax: failed to allocate coords array\n");
        exit(EXIT_FAILURE);
    }
    _softmax_rec(A, axis, 0, coords);
    free(coords);
}

float tensor_get(const Tensor *t, const int *coords) {
    if (!t || !coords) {
        fprintf(stderr, "tensor_get: NULL pointer argument\n");
        exit(EXIT_FAILURE);
    }
    size_t idx = tensor_index(t, coords);
    return t->data[idx];
}

void tensor_set(Tensor *t, const int *coords, float value) {
    if (!t || !coords) {
        fprintf(stderr, "tensor_set: NULL pointer argument\n");
        exit(EXIT_FAILURE);
    }
    size_t idx = tensor_index(t, coords);
    t->data[idx] = value;
}

