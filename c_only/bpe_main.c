#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#include "bpe.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Use:\n");
        fprintf(stderr, "  %s train <corpus.txt> <num_merges>\n", argv[0]);
        fprintf(stderr, "  %s tokenize <merges.txt> <vocab.txt> \"<texto>\"\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Use: %s train <corpus.txt> <num_merges>\n", argv[0]);
            return 1;
        }
        const char *corpus_path = argv[2];
        int num_merges = atoi(argv[3]);
        if (num_merges <= 0) {
            fprintf(stderr, "num_merges must be > 0\n");
            return 1;
        }
        int res = train_bpe(corpus_path, num_merges);
        if (res == 0) {
            printf("Finished training. Generated:\n");
            printf("  - merges.txt\n");
            printf("  - vocab.txt\n");
        }
        return res;
    }
    else if (strcmp(argv[1], "tokenize") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Use: %s tokenize <merges.txt> <vocab.txt> \"<texto>\"\n", argv[0]);
            return 1;
        }
        const char *merges_path = argv[2];
        const char *vocab_path = argv[3];
        const char *texto  = argv[4];

        MergePair *merges;
        int merges_size;
        if (load_merges_file(merges_path, &merges, &merges_size) != 0) {
            return 1;
        }
        
        if (load_vocab_file(vocab_path) != 0) {
            for (int i = 0; i < merges_size; i++) {
                free(merges[i].first);
                free(merges[i].second);
            }
            free(merges);
            return 1;
        }

        tokenize_text(texto, merges, merges_size);

        for (int i = 0; i < merges_size; i++) {
            free(merges[i].first);
            free(merges[i].second);
        }
        free(merges);
        free_vocab_data();
        return 0;
    }
    else {
        fprintf(stderr, "Modo desconhecido: %s\n", argv[1]);
        fprintf(stderr, "Use 'train' ou 'tokenize'\n");
        return 1;
    }
}

float **load_embedding_matrix(const char *filename, int *out_V, int *out_D) {
    int V = 0;
    for (VNode *p = vocab_list; p; p = p->next) {
        V++;
    }
    if (V == 0) {
        fprintf(stderr, "load_embedding_matrix: empty vocab\n");
        return NULL;
    }

    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error opening embeddings.bin: %s\n", filename);
        return NULL;
    }

    struct stat st;
    if (stat(filename, &st) != 0) {
        fprintf(stderr, "Error figuring out size of %s\n", filename);
        fclose(f);
        return NULL;
    }
    long file_size = st.st_size;
    if (file_size % sizeof(float) != 0) {
        fprintf(stderr, "load_embedding_matrix: tamanho inválido (não múltiplo de 4 bytes)\n");
        fclose(f);
        return NULL;
    }
    long total_floats = file_size / sizeof(float);

    if (total_floats % V != 0) {
        fprintf(stderr, "load_embedding_matrix: total_floats (%ld) not divisible by V (%d)\n",
                total_floats, V);
        fclose(f);
        return NULL;
    }
    int D = total_floats / V;

    float *data = malloc(sizeof(float) * total_floats);
    if (!data) {
        fprintf(stderr, "Error malloc at load_embedding_matrix (data)\n");
        fclose(f);
        return NULL;
    }

    size_t read_count = fread(data, sizeof(float), total_floats, f);
    if (read_count != (size_t)total_floats) {
        fprintf(stderr, "Error reading data from %s (red %zu, expected %ld)\n",
                filename, read_count, total_floats);
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    float **embedding_matrix = malloc(sizeof(float*) * V);
    if (!embedding_matrix) {
        fprintf(stderr, "Error malloc at load_embedding_matrix (embedding_matrix)\n");
        free(data);
        return NULL;
    }

    for (int i = 0; i < V; i++) {
        embedding_matrix[i] = data + (size_t)i * D;
    }

    *out_V = V;
    *out_D = D;
    return embedding_matrix;
}

void free_embedding_matrix(float **mat, int vocab_size) {
    if (!mat) return;
    if (mat[0]) {
        free(mat[0]);
    }
    free(mat);
}