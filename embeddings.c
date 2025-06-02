#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static int count_vocab(const char *vocab_path) {
    FILE *f = fopen(vocab_path, "r");
    if (!f) { perror("fopen"); exit(1); }
    int count = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '\n' && line[0] != '\r') count++;
    }
    fclose(f);
    return count;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Use: %s vocab.txt dim embeddings.bin\n", argv[0]);
        return 1;
    }
    const char *vocab_path = argv[1];
    int D = atoi(argv[2]);
    const char *out_path = argv[3];

    int V = count_vocab(vocab_path);
    printf("Generating %d random vectors of %d dims\n", V, D);

    FILE *f = fopen(out_path, "wb");
    if (!f) { perror("fopen"); return 1; }

    srand((unsigned)time(NULL));
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < D; j++) {
            float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
            float u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
            float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            float val = z0 * 0.01f;
            fwrite(&val, sizeof(float), 1, f);
        }
    }
    fclose(f);
    printf("Embeddings generated at %s\n", out_path);
    return 0;
}
