#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "tensor.hu"
#include "utils.h"

__global__ void tensor_add_kernel(const float *A, const float *B, float *out, size_t elems_this_chunk) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_this_chunk)
        out[idx] = A[idx] + B[idx];
}

__global__ void tensor_ones_kernel(float *data, size_t elems_this_chunk) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_this_chunk)
        data[idx] = 1.0f;
}

__global__ void tensor_sub_kernel(const float *A, const float *B, float *out, size_t elems_this_chunk) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_this_chunk)
        out[idx] = A[idx] - B[idx];
}

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

void tensor_add_cuda(Tensor *out, Tensor *A, Tensor *B, size_t chunk_size) {
    if (A->ndim != B->ndim || A->ndim != out->ndim) {
        fprintf(stderr, "tensor_add_cuda: incompatible dimensions (ndim)\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != out->shape[i]) {
            fprintf(stderr, "tensor_add_cuda: diferent shapes in dimension %d (A=%d, B=%d, out=%d)\n", i, A->shape[i], B->shape[i], out->shape[i]);
            exit(EXIT_FAILURE);
        }
    }

    size_t N = A->size;
    size_t num_chunks = (N + chunk_size - 1) / chunk_size;

    size_t bytes_chunk = chunk_size * sizeof(float);
    float *d_A = NULL, *d_B = NULL, *d_out = NULL;
    
    cudaMalloc((void**)&d_A, bytes_chunk);
    cudaMalloc((void**)&d_B, bytes_chunk);
    cudaMalloc((void**)&d_out, bytes_chunk);

    const int threads_per_block = 256;

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t offset = chunk_idx * chunk_size;

        size_t elems_this_chunk = chunk_size;
        if (offset + elems_this_chunk > N) {
            elems_this_chunk = N - offset;
        }
        size_t bytes_this_chunk = elems_this_chunk * sizeof(float);

        cudaMemcpy(d_A, A->data + offset, bytes_this_chunk, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B->data + offset, bytes_this_chunk, cudaMemcpyHostToDevice);

        int blocks_per_grid = (int)((elems_this_chunk + threads_per_block - 1) / threads_per_block);
        tensor_add_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_out, elems_this_chunk);

        cudaGetLastError();
        cudaDeviceSynchronize();

        cudaMemcpy(out->data + offset, d_out, bytes_this_chunk, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
}

void tensor_sub_cuda(Tensor *out, Tensor *A, Tensor *B, size_t chunk_size) {
    if (A->ndim != B->ndim || A->ndim != out->ndim) {
        fprintf(stderr, "tensor_add_cuda: incompatible dimensions (ndim)\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != out->shape[i]) {
            fprintf(stderr, "tensor_add_cuda: diferent shapes in dimension %d (A=%d, B=%d, out=%d)\n", i, A->shape[i], B->shape[i], out->shape[i]);
            exit(EXIT_FAILURE);
        }
    }

    size_t N = A->size;
    size_t num_chunks = (N + chunk_size - 1) / chunk_size;

    size_t bytes_chunk = chunk_size * sizeof(float);
    float *d_A = NULL, *d_B = NULL, *d_out = NULL;
    
    cudaMalloc((void**)&d_A, bytes_chunk);
    cudaMalloc((void**)&d_B, bytes_chunk);
    cudaMalloc((void**)&d_out, bytes_chunk);

    const int threads_per_block = 256;

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t offset = chunk_idx * chunk_size;

        size_t elems_this_chunk = chunk_size;
        if (offset + elems_this_chunk > N) {
            elems_this_chunk = N - offset;
        }
        size_t bytes_this_chunk = elems_this_chunk * sizeof(float);

        cudaMemcpy(d_A, A->data + offset, bytes_this_chunk, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B->data + offset, bytes_this_chunk, cudaMemcpyHostToDevice);

        int blocks_per_grid = (int)((elems_this_chunk + threads_per_block - 1) / threads_per_block);
        tensor_sub_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_out, elems_this_chunk);

        cudaGetLastError();
        cudaDeviceSynchronize();

        cudaMemcpy(out->data + offset, d_out, bytes_this_chunk, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
}


Tensor* tensor_ones_cuda(int ndim, const int *shape, size_t chunk_size) {
    Tensor *t = tensor_new(ndim, shape);
    if (!t) return NULL;

    size_t N = t->size;
    size_t num_chunks = (N + chunk_size - 1) / chunk_size;
    size_t bytes_chunk = chunk_size * sizeof(float);

    float *d_data;
    cudaMalloc((void**)&d_data, bytes_chunk);

    const int threads_per_block = 256;

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t offset = chunk_idx * chunk_size;
        size_t elems_this_chunk = chunk_size;
        if (offset + elems_this_chunk > N) {
            elems_this_chunk = N - offset;
        }
        size_t bytes_this = elems_this_chunk * sizeof(float);

        int blocks_per_grid = (int)((elems_this_chunk + threads_per_block - 1) / threads_per_block);
        tensor_ones_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, elems_this_chunk);
        cudaDeviceSynchronize();
        cudaMemcpy(t->data + offset, d_data, bytes_this, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_data);
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

void tensor_show(Tensor *t) {
    printf("ndim: %d\n", t->ndim);
    printf("size: %zu\n", t->size);

    printf("shape: ");
    __array_print(t->shape, t->ndim, sizeof(int), __int_print);

    printf("stride: ");
    __array_print(t->stride, t->ndim, sizeof(int), __int_print);

    printf("data:\n");
    int *coords = (int*)malloc(t->ndim * sizeof(int));
    if (!coords) {
        fprintf(stderr, "Error allocating coords\n");
        return;
    }
    __tensor_print_recursive(t, 0, coords);
    printf("\n");
    free(coords);
}

int main(void) {
    const int shape[2] = {2, 2};

    Tensor *A = tensor_ones_cuda(2, shape);
    Tensor *B = tensor_ones_cuda(2, shape);

    Tensor *out = tensor_new(2, shape);
    tensor_sub_cuda(out, A, B, 256);

    tensor_show(out);
    
    return 0;
}
