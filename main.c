#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#include "tensor.h"
#include "linear.h"
#include "utils.h"

#define EPOCHS 1

#define LR     1e-3f
#define BETA1  0.9f
#define BETA2  0.999f
#define EPS    1e-8f

static Tensor* softmax_and_grad(const Tensor *logits, int target_id) {
    int V = logits->shape[1];
    Tensor *probs = tensor_new(2, (int[]){1, V});
    if (!probs) {
        fprintf(stderr, " Error allocating probs at softmax_and_grad\n");
        exit(1);
    }
    
    for (int j = 0; j < V; ++j) {
        probs->data[j] = logits->data[j];
    }
    
    float maxv = -INFINITY;
    for (int j = 0; j < V; ++j) {
        if (probs->data[j] > maxv) maxv = probs->data[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < V; ++j) {
        probs->data[j] = expf(probs->data[j] - maxv);
        sum += probs->data[j];
    }
    for (int j = 0; j < V; ++j) {
        probs->data[j] /= sum;
    }
    
    Tensor *dY = tensor_new(2, (int[]){1, V});
    if (!dY) {
        fprintf(stderr, " Error allocating dY at softmax_and_grad\n");
        exit(1);
    }
    for (int j = 0; j < V; ++j) {
        float p = probs->data[j];
        dY->data[j] = p - (j == target_id ? 1.0f : 0.0f);
    }
    tensor_free(probs);
    return dY;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Use: %s <merges.txt> <vocab.txt> <embeddings.bin> <corpus.txt>\n", argv[0]);
        return 1;
    }
    const char *merges_path = argv[1];
    const char *vocab_path  = argv[2];
    const char *embed_path  = argv[3];
    const char *corpus_path = argv[4];

    MergePair *merges;
    int merges_size;
    if (load_merges_file(merges_path, &merges, &merges_size) != 0) {
        fprintf(stderr, "Failed to load merges.txt\n");
        return 1;
    }

    if (load_vocab_file(vocab_path) != 0) {
        fprintf(stderr, "Failed to load vocab.txt\n");
        for (int i = 0; i < merges_size; i++) {
            free(merges[i].first);
            free(merges[i].second);
        }
        free(merges);
        return 1;
    }

    int V, D;
    float **embedding_matrix = load_embedding_matrix(embed_path, &V, &D);
    if (!embedding_matrix) {
        fprintf(stderr, "Failed to load embeddings.bin\n");
        for (int i = 0; i < merges_size; i++) {
            free(merges[i].first);
            free(merges[i].second);
        }
        free(merges);
        free_vocab_data();
        return 1;
    }
    printf("Vocab: %d subwords, Embedding dim: %d\n", V, D);

    Linear *lm = linear_new(D, V);

    FILE *fc = fopen(corpus_path, "r");
    if (!fc) {
        fprintf(stderr, " Error opening corpus: %s\n", corpus_path);
        goto cleanup;
    }

    char line[MAX_LINE];

    int max_ids = 4096;
    int *ids = malloc(sizeof(int) * max_ids);
    if (!ids) {
        fprintf(stderr, " Error malloc ids\n");
        goto cleanup;
    }

    Tensor *x_input = tensor_new(2, (int[]){1, D});
    Tensor *logits  = NULL;
    Tensor *dY      = NULL;
    Tensor *dX      = NULL;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("=== Epoch %d ===\n", epoch + 1);
        fseek(fc, 0, SEEK_SET);

        long step = 0;
        while (fgets(line, sizeof(line), fc)) {
            size_t L = strlen(line);
            if (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) {
                line[L-1] = '\0';
            }
            if (line[0] == '\0') continue;

            char *text_copy = strdup(line);
            const char *delim = " \t\r\n";
            char *word = strtok(text_copy, delim);
            int n_ids = 0;

            while (word) {
                int wlen = (int)strlen(word);
                char **symbols = malloc(sizeof(char*) * (wlen + 1));
                int sym_sz = 0;
                for (int i = 0; i < wlen; i++) {
                    symbols[sym_sz] = malloc(2);
                    symbols[sym_sz][0] = word[i];
                    symbols[sym_sz][1] = '\0';
                    sym_sz++;
                }
                symbols[sym_sz] = strdup("</w>");
                sym_sz++;

                while (1) {
                    int best_rank = INT_MAX;
                    int best_i    = -1;
                    for (int i = 0; i < sym_sz - 1; i++) {
                        for (int m = 0; m < merges_size; m++) {
                            if (strcmp(symbols[i], merges[m].first) == 0 &&
                                strcmp(symbols[i+1], merges[m].second) == 0) {
                                if (merges[m].rank < best_rank) {
                                    best_rank = merges[m].rank;
                                    best_i    = i;
                                }
                                if (best_rank == 0) break;
                            }
                        }
                        if (best_rank == 0) break;
                    }
                    if (best_i < 0) break;

                    size_t la = strlen(symbols[best_i]);
                    size_t lb = strlen(symbols[best_i+1]);
                    char *merged = malloc(la + lb + 1);
                    strcpy(merged, symbols[best_i]);
                    strcat(merged, symbols[best_i+1]);

                    free(symbols[best_i]);
                    free(symbols[best_i+1]);
                    symbols[best_i] = merged;
                    for (int j = best_i + 1; j < sym_sz - 1; j++) {
                        symbols[j] = symbols[j + 1];
                    }
                    sym_sz--;
                }

                for (int i = 0; i < sym_sz; i++) {
                    int id = vocab_get_id(symbols[i]);
                    if (id < 0) {
                        id = vocab_get_id("<unk>");
                        if (id < 0) id = 0;
                    }
                    if (n_ids < max_ids) {
                        ids[n_ids++] = id;
                    }
                    free(symbols[i]);
                }
                free(symbols);
                word = strtok(NULL, delim);
            }
            free(text_copy);

            for (int i = 0; i < n_ids - 1; i++) {
                int id_in  = ids[i];
                int id_out = ids[i+1];

                memcpy(x_input->data, embedding_matrix[id_in], sizeof(float) * D);

                logits = linear_forward(lm, x_input);
                dY = softmax_and_grad(logits, id_out);
                dX = linear_backward(lm, dY);
                linear_update_adam(lm, LR, BETA1, BETA2, EPS);

                tensor_free(logits);
                tensor_free(dY);
                tensor_free(dX);

                step++;
                if (step % 10000 == 0) {
                    printf("Steps: %ld\r", step);
                    fflush(stdout);
                }
            }
        }
        printf("\nEnd of epoch %d, total steps: %ld\n", epoch + 1, step);
    }

    fclose(fc);
    tensor_free(x_input);
    free(ids);

cleanup:
    for (int i = 0; i < merges_size; i++) {
        free(merges[i].first);
        free(merges[i].second);
    }
    free(merges);
    free_vocab_data();

    free_embedding_matrix(embedding_matrix, V);

    linear_free(lm);

    return 0;
}
