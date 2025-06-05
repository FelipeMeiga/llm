#include <stdio.h>
#include <stdlib.h>

void __float_print(void *elem) {
    printf("%.2f ", *(float*)elem);
}

void __int_print(void *elem) {
    printf("%d ", *(int*)elem);
}

void __array_print(void *array, size_t length, size_t elem_size, void (*printer)(void *elem)) {
    char *ptr = (char*)array;
    for (size_t i = 0; i < length; ++i) {
        printer(ptr + i * elem_size);
    }
    printf("\n");
}
