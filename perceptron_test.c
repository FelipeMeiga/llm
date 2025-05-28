#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "linear.h"
#include "tensor.h"

int main(void) {
    int batch = 4;
    int in_dim = 4;
    int hidden_dim = 4;
    int out_dim = 1;

    float lr = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;

    int epochs = 400;

    float X_data[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    float Y_data[4][1] = {{0},{1},{1},{0}};

    int shapeX[2] = { batch, in_dim };
    Tensor *X = tensor_new(2, shapeX);
    for (int i = 0; i < batch; ++i)
        for (int j = 0; j < in_dim; ++j)
            X->data[i*in_dim + j] = X_data[i][j];

    int shapeY[2] = { batch, out_dim };
    Tensor *Y = tensor_new(2, shapeY);
    for (int i = 0; i < batch; ++i)
        Y->data[i*out_dim] = Y_data[i][0];

    // 2->2->1
    Linear *l1 = linear_new(in_dim, hidden_dim);
    Linear *l2 = linear_new(hidden_dim, out_dim);

    for (int e = 1; e <= epochs; ++e) {
        // forward
        Tensor *Y1 = linear_forward(l1, X);
        Tensor *A1 = tensor_reshape(Y1, 2, (int[]){batch, hidden_dim});
        tensor_relu(Y1);
        Tensor *Z2 = linear_forward(l2, Y1);
        tensor_sigmoid(Z2);

        // loss and gradient
        float bce = 0.0f;
        Tensor *dY2 = tensor_new(2, shapeY);
        for (size_t k = 0; k < Z2->size; ++k) {
            float y_true = Y->data[k];
            float y_pred = Z2->data[k];
            bce += - (y_true*logf(y_pred + 1e-8f) + (1.0f-y_true)*logf(1.0f-y_pred + 1e-8f));
            dY2->data[k] = ( -y_true/(y_pred+1e-8f) + (1.0f-y_true)/(1.0f-y_pred+1e-8f) ) / batch;
        }
        bce /= batch;

        // backward
        Tensor *dZ2 = tensor_sigmoid_backward(Z2, dY2);
        Tensor *dA1 = linear_backward(l2, dZ2);
        Tensor *dY1 = tensor_relu_backward(A1, dA1);
        Tensor *dX  = linear_backward(l1, dY1);

        // Adam updates 
        linear_update_adam(l2, lr, beta1, beta2, eps);
        linear_update_adam(l1, lr, beta1, beta2, eps);

        tensor_free(Y1);
        tensor_free(A1);
        tensor_free(Z2);
        tensor_free(dY2);
        tensor_free(dZ2);
        tensor_free(dA1);
        tensor_free(dY1);
        tensor_free(dX);

        if (e % 1000 == 0) {
            printf("epoch %5d, BCE = %.6f\n", e, bce);
        }
    }

    printf("\n=== Teste final (y_pred) ===\n");
    Tensor *Y1 = linear_forward(l1, X);
    tensor_relu(Y1);
    Tensor *Z2 = linear_forward(l2, Y1);
    tensor_sigmoid(Z2);
    tensor_show(Z2);

    tensor_free(X);
    tensor_free(Y);
    tensor_free(Y1);
    tensor_free(Z2);
    linear_free(l1);
    linear_free(l2);

    return 0;
}
