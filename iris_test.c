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

    const int MAX_SAMPLES = 150;
    const int FEATURES = 4;
    const int CLASSES = 3;

    float iris_X[MAX_SAMPLES][FEATURES];
    int iris_y[MAX_SAMPLES];
    int n = load_iris_csv("iris-dataset/Iris.csv", iris_X, iris_y, MAX_SAMPLES);
    if (n < 0) {
        fprintf(stderr, "Error reading iris.csv\n");
        return 1;
    }
    printf("Saved %d lines of Iris.\n", n);

    const int N = n;
    const int F = FEATURES;
    const int C = CLASSES;

    Tensor *X = tensor_new(2, (int[]){N, F});
    Tensor *Y = tensor_new(2, (int[]){N, C});
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < F; ++j) {
            X->data[i*F + j] = iris_X[i][j];
        }
        for (int k = 0; k < C; ++k) {
            Y->data[i*C + k] = (iris_y[i] == k ? 1.0f : 0.0f);
        }
    }

    // 4->16->3
    Linear *l1 = linear_new(F,  16);
    Linear *l2 = linear_new(16, C);

    float lr = 0.005f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps   = 1e-8f;
    int epochs = 5000;

    for (int e = 1; e <= epochs; ++e) {
        Tensor *Y1 = linear_forward(l1, X); // [N×16]
        tensor_relu(Y1);
        Tensor *Z2 = linear_forward(l2, Y1); // [N×3]
        tensor_softmax(Z2, 1); // axis=1

        // loss CE and gradient dZ
        float loss = 0.0f;
        Tensor *dZ = tensor_new(2, (int[]){N, C});
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < C; ++k) {
                float p = Z2->data[i*C + k];
                float t = Y->data[i*C + k];
                loss += -t * logf(p + 1e-8f);
                dZ->data[i*C + k] = (p - t) / N;
            }
        }
        loss /= N;

        // backward
        Tensor *dA1 = linear_backward(l2, dZ); // [N×16]
        Tensor *dX  = tensor_relu_backward(Y1, dA1); // [N×16]
        Tensor *d_in= linear_backward(l1, dX); // [N×4]

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
            printf("epoch %4d/%d   loss = %.4f\n", e, epochs, loss);
        }
    }

    Tensor *Y1 = linear_forward(l1, X);
    tensor_relu(Y1);
    Tensor *Z2 = linear_forward(l2, Y1);
    tensor_softmax(Z2, 1);

    int correct = 0;
    for (int i = 0; i < N; ++i) {
        int pred = 0;
        float best = Z2->data[i*C];
        for (int k = 1; k < C; ++k) {
            float v = Z2->data[i*C + k];
            if (v > best) {
                best = v;
                pred = k;
            }
        }
        if (pred == iris_y[i]) correct++;
    }

    printf("Training accuracy: %.2f%%\n", 100.0f * correct / N);

    tensor_free(X);
    tensor_free(Y);
    tensor_free(Y1);
    tensor_free(Z2);
    linear_free(l1);
    linear_free(l2);

    return 0;
}
