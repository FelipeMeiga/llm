#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linear.hu"
#include "tensor.hu"

__global__ void add_bias_kernel(float *Y, const float *b, int B, int D, int stride0, int stride1) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * D;
    if (idx >= total) return;

    int i = idx / D;
    int j = idx % D;
    size_t offset = (size_t)i * stride0 + (size_t)j * stride1;
    Y[offset] += b[j];
}

__global__ void bias_grad_kernel(const float *dY, float *db, int B, int D, int stride0, int stride1) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float sum = 0.0f;
    
    for (int i = 0; i < B; ++i)
        sum += dY[i * stride0 + j * stride1];

    db[j] = sum;
}

static void add_bias_cuda(Tensor *Y, const Tensor *b) {
    int B = Y->shape[0];
    int D = Y->shape[1];
    int threads = 256;
    int blocks = (B * D + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(Y->data, b->data, B, D, Y->stride[0], Y->stride[1]);
    cudaDeviceSynchronize();
}

static void compute_bias_grad_cuda(const Tensor *dY, Tensor *db) {
    int B = dY->shape[0];
    int D = dY->shape[1];
    int threads = 256;
    int blocks = (D + threads - 1) / threads;
    size_t dy_bytes = (size_t)B * D * sizeof(float);
    size_t db_bytes = (size_t)D * sizeof(float);

    float *d_dY, *d_db;
    cudaMalloc(&d_dY, dy_bytes);
    cudaMalloc(&d_db, db_bytes);
    cudaMemcpy(d_dY, dY->data, dy_bytes, cudaMemcpyHostToDevice);

    bias_grad_kernel<<<blocks, threads>>>(d_dY, d_db, B, D, dY->stride[0], dY->stride[1]);
    cudaDeviceSynchronize();

    cudaMemcpy(db->data, d_db, db_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dY);
    cudaFree(d_db);
}

Linear* linear_new(int in_dim, int out_dim) {
    Linear *lin = (Linear*)malloc(sizeof(Linear));
    if (!lin) {
        fprintf(stderr, "linear_new: failed to allocate Linear struct\n");
        exit(EXIT_FAILURE);
    }

    lin->in_dim  = in_dim;
    lin->out_dim = out_dim;

    int shapeW[2] = { in_dim, out_dim };
    lin->W = tensor_rand_cuda(2, shapeW);
    if (!lin->W) {
        fprintf(stderr, "linear_new: failed to allocate weight tensor W\n");
        exit(EXIT_FAILURE);
    }

    int shapeB[1] = { out_dim };
    lin->b = tensor_zeros_cuda(1, shapeB);
    if (!lin->b) {
        fprintf(stderr, "linear_new: failed to allocate bias tensor b\n");
        exit(EXIT_FAILURE);
    }

    lin->dW = tensor_zeros_cuda(2, shapeW);
    if (!lin->dW) {
        fprintf(stderr, "linear_new: failed to allocate gradient tensor dW\n");
        exit(EXIT_FAILURE);
    }

    lin->db = tensor_zeros_cuda(1, shapeB);
    if (!lin->db) {
        fprintf(stderr, "linear_new: failed to allocate gradient tensor db\n");
        exit(EXIT_FAILURE);
    }

    lin->X = NULL;
    lin->mW = tensor_zeros_cuda(2, shapeW);
    lin->vW = tensor_zeros_cuda(2, shapeW);
    lin->mb = tensor_zeros_cuda(1, shapeB);
    lin->vb = tensor_zeros_cuda(1, shapeB);
    lin->t  = 0;

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

Tensor* linear_forward_cuda(Linear *lin, const Tensor *X, size_t chunk_size) {
    if (X->ndim != 2 || X->shape[1] != lin->in_dim) {
        fprintf(stderr,"linear_forward_cuda: expected X shape [B,%d], got [%d,%d]\n",lin->in_dim, X->shape[0], X->shape[1]);
        exit(EXIT_FAILURE);
    }

    lin->X = (Tensor*)X;

    int B = X->shape[0];
    int shapeY[2] = { B, lin->out_dim };
    Tensor *Y = tensor_new(2, shapeY);

    tensor_matmul_cuda(Y, X, lin->W, chunk_size);

    add_bias_cuda(Y, lin->b);

    return Y;
}

Tensor* linear_backward_cuda(Linear *lin, const Tensor *dY, size_t chunk_size) {
    if (!lin->X) {
        fprintf(stderr, "linear_backward_cuda: no saved input\n");
        exit(EXIT_FAILURE);
    }

    const Tensor *X = lin->X;
    int B    = X->shape[0];
    int Din  = lin->in_dim;
    int Dout = lin->out_dim;

    int shapeW[2] = { Din, Dout };
    Tensor *new_dW = tensor_new(2, shapeW);
    
    int shapeXt[2] = { Din, B };
    Tensor *Xt = tensor_new(2, shapeXt);
    tensor_transpose_cuda(X, Xt, 0, 1);
    
    tensor_matmul_cuda(new_dW, Xt, dY, chunk_size);
    tensor_free(Xt);
    tensor_free(lin->dW);
    lin->dW = new_dW;

    compute_bias_grad_cuda(dY, lin->db);

    int shapeWt[2] = { Dout, Din };
    Tensor *Wt = tensor_new(2, shapeWt);
    tensor_transpose_cuda(lin->W, Wt, 0, 1);

    int shapeDX[2] = { B, Din };
    Tensor *dX = tensor_new(2, shapeDX);
    tensor_matmul_cuda(dX, dY, Wt, chunk_size);
    tensor_free(Wt);

    return dX;
}

