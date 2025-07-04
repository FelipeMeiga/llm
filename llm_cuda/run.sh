#!/usr/bin/env bash
nvcc -o main iris_test.cu linear.cu tensor.cu utils.c
./main
