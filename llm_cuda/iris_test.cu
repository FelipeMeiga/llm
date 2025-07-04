#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "linear.hu"
#include "tensor.hu"

void linear_update_adam(Linear *lin, float lr, float beta1, float beta2, float eps) {
    lin->t += 1;
    float b1t = powf(beta1, lin->t);
    float b2t = powf(beta2, lin->t);

    for (size_t i = 0; i < lin->W->size; ++i) {
        float g  = lin->dW->data[i];
        float m  = lin->mW->data[i] = beta1*lin->mW->data[i] + (1-beta1)*g;
        float v  = lin->vW->data[i] = beta2*lin->vW->data[i] + (1-beta2)*g*g;
        float m_hat = m / (1 - b1t);
        float v_hat = v / (1 - b2t);
        lin->W->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }

    for (size_t i = 0; i < lin->b->size; ++i) {
        float g  = lin->db->data[i];
        float m  = lin->mb->data[i] = beta1*lin->mb->data[i] + (1-beta1)*g;
        float v  = lin->vb->data[i] = beta2*lin->vb->data[i] + (1-beta2)*g*g;
        float m_hat = m / (1 - b1t);
        float v_hat = v / (1 - b2t);
        lin->b->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
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

    int *coords = (int*)malloc(A->ndim * sizeof(int));
    if (!coords) {
        fprintf(stderr, "tensor_softmax: failed to allocate coords array\n");
        exit(EXIT_FAILURE);
    }
    _softmax_rec(A, axis, 0, coords);
    free(coords);
}

void tensor_dropout(Tensor *A, float p) {
    for (size_t i = 0; i < A->size; ++i) {
        float u = (float)rand() / (float)RAND_MAX;
        if (u < p) {
            A->data[i] = 0.0f;
        } else {
            A->data[i] /= (1.0f - p);
        }
    }
}

int load_iris_csv(const char *path, float iris_X[][4], int iris_y[], int max_samples) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("load_iris_csv: fopen");
        return -1;
    }

    char line[256];
    
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }

    int count = 0;
    while (count < max_samples && fgets(line, sizeof(line), f)) {
        char *tok = strtok(line, ",");
        if (!tok) break;

        tok = strtok(NULL, ",");  
        float sl = atof(tok);

        tok = strtok(NULL, ",");  
        float sw = atof(tok);

        tok = strtok(NULL, ",");  
        float pl = atof(tok);

        tok = strtok(NULL, ",");  
        float pw = atof(tok);

        tok = strtok(NULL, ",\r\n");  
        if (!tok) break;
        
        tok[strcspn(tok, "\r\n")] = 0;

        int label;
        if (strcmp(tok, "Iris-setosa")     == 0) label = 0;
        else if (strcmp(tok, "Iris-versicolor") == 0) label = 1;
        else if (strcmp(tok, "Iris-virginica")  == 0) label = 2;
        else label = -1;

        iris_X[count][0] = sl;
        iris_X[count][1] = sw;
        iris_X[count][2] = pl;
        iris_X[count][3] = pw;
        iris_y[count] = label;

        count++;
    }

    fclose(f);
    return count;
}

int main(void) {
    srand((unsigned)time(NULL));

    const int MAX_SAMPLES = 150, FEATURES = 4, CLASSES = 3;
    float iris_X[MAX_SAMPLES][FEATURES];
    int iris_y[MAX_SAMPLES];
    int n = load_iris_csv("iris-dataset/Iris.csv", iris_X, iris_y, MAX_SAMPLES);
    if (n < 0) {
        fprintf(stderr, "Error reading iris.csv\n");
        return 1;
    }
    printf("Saved %d samples from Iris.\n", n);

    int idx[MAX_SAMPLES];
    for (int i = 0; i < n; ++i) idx[i] = i;
    for (int i = n-1; i > 0; --i) {
        int j = rand() % (i+1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    int train_n = (int)(0.8f * n);
    int val_n   = n - train_n;

    int train[] = {train_n, FEATURES};
    int train2[] = {train_n, CLASSES};
    int val[] = {val_n, FEATURES};
    int val2[] = {val_n, CLASSES};

    Tensor *X_train = tensor_new(2, train);
    Tensor *Y_train = tensor_new(2, train2);
    Tensor *X_val = tensor_new(2, val);
    Tensor *Y_val = tensor_new(2, val2);

    for (int i = 0; i < train_n; ++i) {
        int id = idx[i];
        for (int j = 0; j < FEATURES; ++j)
            X_train->data[i*FEATURES + j] = iris_X[id][j];
        for (int k = 0; k < CLASSES; ++k)
            Y_train->data[i*CLASSES + k] = (iris_y[id] == k ? 1.0f : 0.0f);
    }
    for (int i = 0; i < val_n; ++i) {
        int id = idx[train_n + i];
        for (int j = 0; j < FEATURES; ++j)
            X_val->data[i*FEATURES + j] = iris_X[id][j];
        for (int k = 0; k < CLASSES; ++k)
            Y_val->data[i*CLASSES + k] = (iris_y[id] == k ? 1.0f : 0.0f);
    }

    // 4->16->3
    Linear *l1 = linear_new(FEATURES, 16);
    Linear *l2 = linear_new(16, CLASSES);

    float lr = 0.01f;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int epochs = 2000;

    for (int e = 1; e <= epochs; ++e) {
        // forward train
        Tensor *Y1 = linear_forward_cuda(l1, X_train, 256);
        tensor_relu_cuda(Y1, 256);
        tensor_dropout(Y1, 0.1f);
        Tensor *Z2 = linear_forward_cuda(l2, Y1, 256);
        tensor_softmax(Z2, 1);

        // train loss and dZ
        float train_loss = 0.0f;
        Tensor *dZ = tensor_new(2, train2);
        for (int i = 0; i < train_n; ++i) {
            for (int k = 0; k < CLASSES; ++k) {
                float p = Z2->data[i*CLASSES + k];
                float t = Y_train->data[i*CLASSES + k];
                train_loss += -t * logf(p + 1e-8f);
                dZ->data[i*CLASSES + k] = (p - t) / train_n;
            }
        }
        train_loss /= train_n;

        // backward train
        Tensor *dA1 = linear_backward_cuda(l2, dZ, 256);
        Tensor *dX = tensor_relu_backward_cuda(Y1, dA1, 256);
        Tensor *d_in= linear_backward_cuda(l1, dX, 256);

        // Adam updates
        linear_update_adam(l2, lr, beta1, beta2, eps);
        linear_update_adam(l1, lr, beta1, beta2, eps);

        tensor_free(Y1);
        tensor_free(Z2);
        tensor_free(dZ);
        tensor_free(dA1);
        tensor_free(dX);
        tensor_free(d_in);

        if (e % 200 == 0) {
            // forward val
            Tensor *Y1v = linear_forward_cuda(l1, X_val, 256);
            tensor_relu_cuda(Y1v, 256);
            Tensor *Z2v = linear_forward_cuda(l2, Y1v, 256);
            tensor_softmax(Z2v, 1);

            // loss & accuracy
            float val_loss = 0.0f;
            int   correct  = 0;
            for (int i = 0; i < val_n; ++i) {
                // loss
                for (int k = 0; k < CLASSES; ++k) {
                    float p = Z2v->data[i*CLASSES + k];
                    float t = Y_val->data[i*CLASSES + k];
                    val_loss += -t * logf(p + 1e-8f);
                }
                // accuracy
                int  pred = 0;
                float best = Z2v->data[i*CLASSES + 0];
                for (int k = 1; k < CLASSES; ++k) {
                    float v = Z2v->data[i*CLASSES + k];
                    if (v > best) { best = v; pred = k; }
                }
                if (Y_val->data[i*CLASSES + pred] > 0.5f) correct++;
            }
            val_loss /= val_n;
            float val_acc = 100.0f * correct / val_n;

            printf("epoch %4d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.2f%%\n", e, epochs, train_loss, val_loss, val_acc);

            tensor_free(Y1v);
            tensor_free(Z2v);
        }
    }

    tensor_free(X_train);
    tensor_free(Y_train);
    tensor_free(X_val);
    tensor_free(Y_val);
    linear_free(l1);
    linear_free(l2);

    return 0;
}
