#ifndef BPE_H
#define BPE_H

#define MAX_LINE 4096

typedef struct {
    char **symbols;  // array de char*, cada um é "c", "a", "s", … ou "</w>"
    int len;      // quantos símbolos existem atualmente
    int cap;      // capacidade alocada para symbols[]
} Word;

typedef struct {
    char *first;
    char *second;
    int rank;
} MergePair;

typedef struct {
    char *first;
    char *second;
    int count;
} PairCount;

typedef struct VNode {
    char *token;
    int id;
    struct VNode *next;
} VNode;

static VNode *vocab_list = NULL;

Word *create_word_from_raw(const char *raw);
void free_word(Word *w);
void merge_in_word(Word *w, const char *a, const char *b, const char *merged);

PairCount *count_all_pairs(Word **words, int num_words, int *out_num_pairs);
void free_paircounts(PairCount *pc, int num_pairs);
int find_parcount_index(PairCount *pc, int num_pairs, const char *a, const char *b);
int get_most_frequent_pair(PairCount *pc, int num_pairs, int *out_index);

int train_bpe(const char *corpus_path, int num_merges);
void write_merges(const char *filename, MergePair *merges, int merges_size);
void write_vocab(const char *filename, Word **words, int num_words);

int load_merges_file(const char *filename, MergePair **out_merges, int *out_size);
int load_vocab_file(const char *filename);
int vocab_get_id(const char *token);
void free_vocab_data(void);
void tokenize_text(const char *text, MergePair *merges, int merges_size);

void free_embedding_matrix(float **mat, int vocab_size);
float **load_embedding_matrix(const char *filename, int *out_V, int *out_D);

#endif