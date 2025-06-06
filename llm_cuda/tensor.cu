#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "tensor.hu"
#include "utils.h"

__global__ void tensor_add_kernel(const float *A, const float *B, float *out, size_t elems_this_chunk) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_this_chunk)
        out[idx] = A[idx] + B[idx];
}

__global__ void tensor_sub_kernel(const float *A, const float *B, float *out, size_t elems_this_chunk) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_this_chunk)
        out[idx] = A[idx] - B[idx];
}

__global__ void tensor_scale_kernel(const float *A, const float alpha, float *out, size_t elems_this_chunk) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_this_chunk)
        out[idx] = A[idx] * alpha;
}

__global__ void tensor_mul_kernel(const float *A, const float *B, float *out, size_t elems_this_chunk) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_this_chunk)
        out[idx] = A[idx] * B[idx];
}

__global__ void tensor_ones_kernel(float *data, size_t elems_this_chunk) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_this_chunk)
        data[idx] = 1.0f;
}

__global__ void tensor_zeros_kernel(float *data, size_t elems_this_chunk) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_this_chunk)
        data[idx] = 0.0f;
}

__global__ void tensor_rand_kernel(float *data, size_t elems_this_chunk, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_this_chunk) {
        curandState_t state;
        curand_init(seed, idx, 0, &state);

        data[idx] = curand_uniform(&state);
    }
}

__global__ void matmul_chunk_kernel( const float *A_sub, const float *B_sub, float *C_sub, int m, int p, int n_sub) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n_sub; ++k)
            sum += A_sub[row * n_sub + k] * B_sub[k * p + col];
        C_sub[row * p + col] += sum;
    }
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
        fprintf(stderr, "tensor_sub_cuda: incompatible dimensions (ndim)\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != out->shape[i]) {
            fprintf(stderr, "tensor_sub_cuda: diferent shapes in dimension %d (A=%d, B=%d, out=%d)\n", i, A->shape[i], B->shape[i], out->shape[i]);
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

void tensor_scale_cuda(Tensor *out, Tensor *A, float alpha, size_t chunk_size) {
    if (A->ndim != out->ndim) {
        fprintf(stderr, "tensor_scale_cuda: incompatible dimensions (ndim)\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != out->shape[i]) {
            fprintf(stderr, "tensor_scale_cuda: diferent shapes in dimension %d (A=%d, out=%d)\n", i, A->shape[i], out->shape[i]);
            exit(EXIT_FAILURE);
        }
    }

    size_t N = A->size;
    size_t num_chunks = (N + chunk_size - 1) / chunk_size;

    size_t bytes_chunk = chunk_size * sizeof(float);
    float *d_A = NULL, *d_out = NULL;
    
    cudaMalloc((void**)&d_A, bytes_chunk);
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

        int blocks_per_grid = (int)((elems_this_chunk + threads_per_block - 1) / threads_per_block);
        tensor_scale_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, alpha, d_out, elems_this_chunk);

        cudaGetLastError();
        cudaDeviceSynchronize();

        cudaMemcpy(out->data + offset, d_out, bytes_this_chunk, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A);
    cudaFree(d_out);
}

void tensor_mul_cuda(Tensor *out, Tensor *A, Tensor *B, size_t chunk_size) {
    if (A->ndim != B->ndim || A->ndim != out->ndim) {
        fprintf(stderr, "tensor_mul_cuda: incompatible dimensions (ndim)\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != out->shape[i]) {
            fprintf(stderr, "tensor_mul_cuda: diferent shapes in dimension %d (A=%d, B=%d, out=%d)\n", i, A->shape[i], B->shape[i], out->shape[i]);
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
        tensor_mul_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_out, elems_this_chunk);

        cudaGetLastError();
        cudaDeviceSynchronize();

        cudaMemcpy(out->data + offset, d_out, bytes_this_chunk, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
}

void tensor_matmul_cuda(Tensor *out, const Tensor *A, const Tensor *B, size_t chunk_size) {
    if (A->ndim != 2 || B->ndim != 2) {
        fprintf(stderr, "tensor_matmul_cuda_chunked: only 2d tensors are suported (A.ndim=%d, B.ndim=%d)\n", A->ndim, B->ndim);
        exit(EXIT_FAILURE);
    }
    int m = A->shape[0];
    int n = A->shape[1];
    int n2 = B->shape[0];
    int p = B->shape[1];
    if (n != n2) {
        fprintf(stderr, "tensor_matmul_cuda_chunked: incompatible internal dimensions (A.cols=%d, B.rows=%d)\n", n, n2);
        exit(EXIT_FAILURE);
    }
    
    size_t num_chunks = (n + chunk_size - 1) / chunk_size;

    size_t bytes_A_sub = m * chunk_size * sizeof(float);
    size_t bytes_B_sub = chunk_size * p * sizeof(float);
    size_t bytes_out_sub = m * p * sizeof(float);

    float *d_A_sub = NULL;
    float *d_B_sub = NULL;
    float *d_out_sub = NULL;

    cudaMalloc((void**)&d_A_sub, bytes_A_sub);
    cudaMalloc((void**)&d_B_sub, bytes_B_sub);
    cudaMalloc((void**)&d_out_sub, bytes_out_sub);

    const int TILE_DIM = 16;
    dim3 threads_per_block(TILE_DIM, TILE_DIM);
    dim3 blocks_per_grid(
        (p + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM
    );

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int k_start = chunk_idx * chunk_size;
        int n_sub = chunk_size;
        if (k_start + n_sub > n) {
            n_sub = n - k_start;
        }
        
        for (int i = 0; i < m; ++i) {
            const float *host_ptr_A = A->data + (size_t)i * A->stride[0] + (size_t)k_start * A->stride[1];
            
            float *dev_ptr_A = d_A_sub + (size_t)i * n_sub;
            cudaMemcpy(dev_ptr_A, host_ptr_A, n_sub * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        for (int k_local = 0; k_local < n_sub; ++k_local) {
            const float *host_ptr_B = B->data + (size_t)(k_start + k_local) * B->stride[0];
            float *dev_ptr_B = d_B_sub + (size_t)k_local * p;
            cudaMemcpy(dev_ptr_B, host_ptr_B, p * sizeof(float), cudaMemcpyHostToDevice);
        }

        cudaMemset(d_out_sub, 0, bytes_out_sub);

        matmul_chunk_kernel<<<blocks_per_grid, threads_per_block>>>(d_A_sub, d_B_sub, d_out_sub, m, p, n_sub);
        cudaDeviceSynchronize();

        float *h_out_sub = (float*)malloc(bytes_out_sub);
        cudaMemcpy(h_out_sub, d_out_sub, bytes_out_sub, cudaMemcpyDeviceToHost);

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                size_t idx_out = (size_t)i * out->stride[0] + (size_t)j * out->stride[1];
                size_t idx_sub = (size_t)i * p + (size_t)j;
                out->data[idx_out] += h_out_sub[idx_sub];
            }
        }
        free(h_out_sub);
    }

    cudaFree(d_A_sub);
    cudaFree(d_B_sub);
    cudaFree(d_out_sub);
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

Tensor* tensor_zeros_cuda(int ndim, const int *shape, size_t chunk_size) {
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
        tensor_zeros_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, elems_this_chunk);
        cudaDeviceSynchronize();
        cudaMemcpy(t->data + offset, d_data, bytes_this, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_data);
    return t;
}

Tensor* tensor_rand_cuda(int ndim, const int *shape, size_t chunk_size) {
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
        unsigned long long seed = (unsigned long long)time(NULL);
        tensor_rand_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, elems_this_chunk, seed);
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

static void __tensor_print_recursive(const Tensor *t, int dim, int *coords) {
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

void tensor_free(Tensor *t) {
    free(t->shape);
    free(t->stride);
    free(t->data);
    free(t);
}

void cuda_get_info() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total global memory: %lu\n", prop.totalGlobalMem);
        printf("  Compute capability: %d.%d\n",
               prop.major, prop.minor);
        printf("  Number of SMs: %d\n",
               prop.multiProcessorCount);
        printf("  Max threads per block: %d\n",
               prop.maxThreadsPerBlock);
        printf("  Max threads dimensions: x = %d, y = %d, z = %d\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: x = %d, y = %d, z = %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}

// int main(void) {
//     int ndim = 2;
//     int shape[ndim] = {6, 6};

//     Tensor *A = tensor_rand_cuda(ndim, shape);
//     tensor_show(A);

//     Tensor *B = tensor_rand_cuda(ndim, shape);
//     tensor_show(B);

//     Tensor *out = tensor_new(ndim, shape);
//     tensor_matmul_cuda(out, A, B);
//     tensor_show(out);

//     return 0;
// }
