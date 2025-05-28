#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "linear.h"
#include "tensor.h"

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
        fprintf(stderr, "Erroe reading iris.csv\n");
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

    Tensor *X_train = tensor_new(2, (int[]){train_n, FEATURES});
    Tensor *Y_train = tensor_new(2, (int[]){train_n, CLASSES});
    Tensor *X_val = tensor_new(2, (int[]){val_n, FEATURES});
    Tensor *Y_val = tensor_new(2, (int[]){val_n, CLASSES});

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

    float lr = 0.005f;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int epochs = 2000;

    for (int e = 1; e <= epochs; ++e) {
        // forward train
        Tensor *Y1 = linear_forward(l1, X_train);
        tensor_relu(Y1);
        Tensor *Z2 = linear_forward(l2, Y1);
        tensor_softmax(Z2, 1);

        // train loss and dZ
        float train_loss = 0.0f;
        Tensor *dZ = tensor_new(2, (int[]){train_n, CLASSES});
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
        Tensor *dA1 = linear_backward(l2, dZ);
        Tensor *dX = tensor_relu_backward(Y1, dA1);
        Tensor *d_in= linear_backward(l1, dX);

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
            Tensor *Y1v = linear_forward(l1, X_val);
            tensor_relu(Y1v);
            Tensor *Z2v = linear_forward(l2, Y1v);
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
