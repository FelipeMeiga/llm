#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

void __int_print(void *elem);
void __float_print(void *elem);
void __array_print(void *array, size_t length, size_t elem_size, void (*printer)(void *elem));

#ifdef __cplusplus
}
#endif

#endif
