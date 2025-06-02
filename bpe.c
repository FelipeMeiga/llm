#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>

#include "bpe.h"

Word *create_word_from_raw(const char *raw) {
    int wlen = (int)strlen(raw);

    int cap = wlen + 1;
    Word *w = (Word*)malloc(sizeof(Word));
    if (!w) {
        fprintf(stderr, "Error malloc at create_word_from_raw\n");
        exit(1);
    }
    w->symbols = (char**)malloc(sizeof(char*) * cap);
    if (!w->symbols) {
        fprintf(stderr, "Error malloc at create_word_from_raw (symbols)\n");
        exit(1);
    }
    w->len = 0;
    w->cap = cap;

    for (int i = 0; i < wlen; i++) {
        w->symbols[w->len] = (char*)malloc(2);
        if (!w->symbols[w->len]) {
            fprintf(stderr, "Error malloc at create_word_from_raw (char)\n");
            exit(1);
        }
        w->symbols[w->len][0] = raw[i];
        w->symbols[w->len][1] = '\0';
        w->len++;
    }
    
    w->symbols[w->len] = strdup(" ");
    if (!w->symbols[w->len]) {
        fprintf(stderr, "Error strdup at create_word_from_raw\n");
        exit(1);
    }
    w->len++;

    return w;
}

void free_word(Word *w) {
    if (!w) return;
    for (int i = 0; i < w->len; i++) {
        free(w->symbols[i]);
    }
    free(w->symbols);
    free(w);
}

void merge_in_word(Word *w, const char *a, const char *b, const char *merged) {
    for (int i = 0; i < w->len - 1; i++) {
        if (strcmp(w->symbols[i], a) == 0 && strcmp(w->symbols[i+1], b) == 0) {
            free(w->symbols[i]);
            free(w->symbols[i+1]);
            w->symbols[i] = strdup(merged);
            
            for (int j = i + 1; j < w->len - 1; j++) {
                w->symbols[j] = w->symbols[j + 1];
            }
            w->len--;
            return;
        }
    }
}

PairCount *count_all_pairs(Word **words, int num_words, int *out_num_pairs) {
    PairCount *pc      = NULL;
    int        pc_size = 0;

    for (int w = 0; w < num_words; w++) {
        Word *word = words[w];
        for (int i = 0; i < word->len - 1; i++) {
            const char *a = word->symbols[i];
            const char *b = word->symbols[i+1];
            int idx = find_parcount_index(pc, pc_size, a, b);
            if (idx >= 0) {
                pc[idx].count++;
            } else {
                PairCount *tmp = realloc(pc, sizeof(PairCount)*(pc_size+1));
                if (!tmp) {
                    fprintf(stderr, "Not enough memory at count_all_pairs\n");
                    free_paircounts(pc, pc_size);
                    *out_num_pairs = 0;
                    return NULL;
                }
                pc = tmp;
                pc[pc_size].first = strdup(a);
                pc[pc_size].second = strdup(b);
                pc[pc_size].count = 1;
                pc_size++;
            }
        }
    }
    *out_num_pairs = pc_size;
    return pc;
}

void free_paircounts(PairCount *pc, int num_pairs) {
    if (!pc) return;
    for (int i = 0; i < num_pairs; i++) {
        free(pc[i].first);
        free(pc[i].second);
    }
    free(pc);
}

int find_parcount_index(PairCount *pc, int num_pairs, const char *a, const char *b) {
    for (int i = 0; i < num_pairs; i++) {
        if (strcmp(pc[i].first, a) == 0 && strcmp(pc[i].second, b) == 0) {
            return i;
        }
    }
    return -1;
}

int get_most_frequent_pair(PairCount *pc, int num_pairs, int *out_index) {
    if (num_pairs <= 0) {
        *out_index = -1;
        return -1;
    }
    int best = 0;
    for (int i = 1; i < num_pairs; i++) {
        if (pc[i].count > pc[best].count) {
            best = i;
        }
    }
    *out_index = best;
    return pc[best].count;
}

int train_bpe(const char *corpus_path, int num_merges) {
    FILE *f = fopen(corpus_path, "r");
    if (!f) {
        fprintf(stderr, "Error: couldnt open corpus: %s\n", corpus_path);
        return -1;
    }

    Word **words    = NULL;
    int num_words = 0;
    int cap_words = 0;

    char line[MAX_LINE];
    while (fgets(line, sizeof(line), f)) {
        size_t L = strlen(line);
        if (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) {
            line[L-1] = '\0';
        }
        
        char *token = strtok(line, " \t\r\n");
        while (token) {
            if (num_words >= cap_words) {
                int newcap = (cap_words == 0 ? 128 : cap_words * 2);
                Word **tmp = realloc(words, sizeof(Word*) * newcap);
                if (!tmp) {
                    fprintf(stderr, "Error creating memory for words array\n");
                    fclose(f);
                    for (int i = 0; i < num_words; i++) free_word(words[i]);
                    free(words);
                    return -1;
                }
                words = tmp;
                cap_words = newcap;
            }
            
            words[num_words] = create_word_from_raw(token);
            num_words++;
            token = strtok(NULL, " \t\r\n");
        }
    }
    fclose(f);
    if (num_words == 0) {
        fprintf(stderr, "Warning: empty corpus.\n");
        return -1;
    }

    MergePair *merges = (MergePair*)malloc(sizeof(MergePair) * num_merges);
    int merges_sz = 0;

    for (int m = 0; m < num_merges; m++) {
        int    num_pairs = 0;
        PairCount *pc = count_all_pairs(words, num_words, &num_pairs);
        if (!pc || num_pairs == 0) {
            if (pc) free_paircounts(pc, num_pairs);
            break;
        }
        
        int best_idx;
        int best_count = get_most_frequent_pair(pc, num_pairs, &best_idx);
        if (best_count < 1) {
            free_paircounts(pc, num_pairs);
            break;
        }

        merges[merges_sz].first  = strdup(pc[best_idx].first);
        merges[merges_sz].second = strdup(pc[best_idx].second);
        merges[merges_sz].rank   = merges_sz;
        merges_sz++;

        size_t la = strlen(pc[best_idx].first);
        size_t lb = strlen(pc[best_idx].second);
        char *merged = (char*)malloc(la + lb + 1);
        if (!merged) {
            fprintf(stderr, "Error malloc at merged\n");
            exit(1);
        }
        strcpy(merged, pc[best_idx].first);
        strcat(merged, pc[best_idx].second);

        for (int w = 0; w < num_words; w++) {
            while (1) {
                Word *wd = words[w];
                int found = 0;
                for (int i = 0; i < wd->len - 1; i++) {
                    if (strcmp(wd->symbols[i], pc[best_idx].first) == 0 &&
                        strcmp(wd->symbols[i+1], pc[best_idx].second) == 0) {
                        found = 1;
                        break;
                    }
                }
                if (!found) break;
                merge_in_word(wd, pc[best_idx].first, pc[best_idx].second, merged);
            }
        }

        free(merged);
        free_paircounts(pc, num_pairs);
    }

    write_merges("merges.txt", merges, merges_sz);
    write_vocab ("vocab.txt", words, num_words);

    for (int i = 0; i < merges_sz; i++) {
        free(merges[i].first);
        free(merges[i].second);
    }
    free(merges);

    for (int i = 0; i < num_words; i++) {
        free_word(words[i]);
    }
    free(words);

    return 0;
}

void write_merges(const char *filename, MergePair *merges, int merges_size) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error opening %s\n", filename);
        return;
    }
    for (int i = 0; i < merges_size; i++) {
        fprintf(f, "%s %s\n", merges[i].first, merges[i].second);
    }
    fclose(f);
}

void write_vocab(const char *filename, Word **words, int num_words) {
    char **unique_syms = NULL;
    int    uniq_sz = 0;

    for (int w = 0; w < num_words; w++) {
        Word *wd = words[w];
        for (int i = 0; i < wd->len; i++) {
            const char *sym = wd->symbols[i];
            int exists = 0;
            for (int j = 0; j < uniq_sz; j++) {
                if (strcmp(unique_syms[j], sym) == 0) {
                    exists = 1;
                    break;
                }
            }
            if (!exists) {
                char **tmp = realloc(unique_syms, sizeof(char*)*(uniq_sz+1));
                if (!tmp) {
                    fprintf(stderr, "Error realloc at write_vocab\n");
                    exit(1);
                }
                unique_syms = tmp;
                unique_syms[uniq_sz++] = strdup(sym);
            }
        }
    }
    
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error opening %s\n", filename);
        for (int i = 0; i < uniq_sz; i++) free(unique_syms[i]);
        free(unique_syms);
        return;
    }
    for (int i = 0; i < uniq_sz; i++) {
        fprintf(f, "%s %d\n", unique_syms[i], i);
    }
    fclose(f);

    for (int i = 0; i < uniq_sz; i++) free(unique_syms[i]);
    free(unique_syms);
}

int load_merges_file(const char *filename, MergePair **out_merges, int *out_size) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening merges.txt: %s\n", filename);
        return -1;
    }
    MergePair *merges = NULL;
    int merges_sz = 0;

    char line[MAX_LINE];
    while (fgets(line, sizeof(line), f)) {
        size_t L = strlen(line);
        if (L > 0 && (line[L-1]=='\n' || line[L-1]=='\r')) {
            line[L-1] = '\0';
        }
        char tok1[256], tok2[256];
        if (sscanf(line, "%255s %255s", tok1, tok2) != 2) continue;
        MergePair *tmp = realloc(merges, sizeof(MergePair)*(merges_sz+1));
        if (!tmp) {
            fprintf(stderr, "Error memory loading merges\n");
            exit(1);
        }
        merges = tmp;
        merges[merges_sz].first  = strdup(tok1);
        merges[merges_sz].second = strdup(tok2);
        merges[merges_sz].rank   = merges_sz;
        merges_sz++;
    }
    fclose(f);
    *out_merges = merges;
    *out_size   = merges_sz;
    return 0;
}

int load_vocab_file(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening vocab.txt: %s\n", filename);
        return -1;
    }
    char line[MAX_LINE];
    while (fgets(line, sizeof(line), f)) {
        char tok[256];
        int  idx;
        if (sscanf(line, "%255s %d", tok, &idx) != 2) continue;
        VNode *n = malloc(sizeof(VNode));
        if (!n) {
            fprintf(stderr, "Error malloc at load_vocab_file\n");
            exit(1);
        }
        n->token = strdup(tok);
        n->id    = idx;
        n->next  = vocab_list;
        vocab_list = n;
    }
    fclose(f);
    return 0;
}

int vocab_get_id(const char *token) {
    for (VNode *p = vocab_list; p; p = p->next) {
        if (strcmp(p->token, token) == 0) {
            return p->id;
        }
    }
    return -1;
}

void free_vocab_data(void) {
    VNode *p = vocab_list;
    while (p) {
        VNode *tmp = p->next;
        free(p->token);
        free(p);
        p = tmp;
    }
    vocab_list = NULL;
}

void tokenize_word(const char *word, MergePair *merges, int merges_size) {
    int wlen = (int)strlen(word);
    char **symbols = malloc(sizeof(char*) * (wlen + 1));
    if (!symbols) {
        fprintf(stderr, "Error malloc at tokenize_word\n");
        exit(1);
    }
    int sym_sz  = 0;

    for (int i = 0; i < wlen; i++) {
        symbols[sym_sz] = malloc(2);
        if (!symbols[sym_sz]) {
            fprintf(stderr, "Error malloc at tokenize_word (char)\n");
            exit(1);
        }
        symbols[sym_sz][0] = word[i];
        symbols[sym_sz][1] = '\0';
        sym_sz++;
    }
    symbols[sym_sz] = strdup(" ");
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
        if (!merged) {
            fprintf(stderr, "Error malloc at tokenize_word (merged)\n");
            exit(1);
        }
        strcpy(merged, symbols[best_i]);
        strcat(merged, symbols[best_i+1]);

        free(symbols[best_i]);
        free(symbols[best_i+1]);
        symbols[best_i] = merged;
        // shift
        for (int j = best_i + 1; j < sym_sz - 1; j++) {
            symbols[j] = symbols[j + 1];
        }
        sym_sz--;
    }

    for (int i = 0; i < sym_sz; i++) {
        int id = vocab_get_id(symbols[i]);
        if (id < 0) {
            id = vocab_get_id("<unk>");
            if (id < 0) {
                printf("[%s:-1] ", symbols[i]);
            } else {
                printf("[%s:%d] ", symbols[i], id);
            }
        } else {
            printf("[%s:%d] ", symbols[i], id);
        }
        free(symbols[i]);
    }
    free(symbols);
}

void tokenize_text(const char *text, MergePair *merges, int merges_size) {
    char *copy = strdup(text);
    if (!copy) {
        fprintf(stderr, "Error de malloc em tokenize_text\n");
        exit(1);
    }
    const char *delim = " \t\r\n";
    char *token = strtok(copy, delim);
    while (token) {
        tokenize_word(token, merges, merges_size);
        printf("\n");
        token = strtok(NULL, delim);
    }
    free(copy);
}

float **load_embedding_matrix(const char *filename, int *out_V, int *out_D) {
    int V = 1;
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
        fprintf(stderr, "load_embedding_matrix: invalid size (not multiple of 4 bytes)\n");
        fclose(f);
        return NULL;
    }
    long total_floats = file_size / sizeof(float);

    if (total_floats % V != 0) {
        fprintf(stderr, "load_embedding_matrix: total_floats (%ld) not divisible by V (%d)\n", total_floats, V);
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
        fprintf(stderr, "Error reading data from %s (red %zu, expected %ld)\n", filename, read_count, total_floats);
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
