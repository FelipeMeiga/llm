#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"

Tensor *scaled_dot_product_attention(const Tensor *Q, const Tensor *K, const Tensor *V);

#endif