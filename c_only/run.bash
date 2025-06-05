bash run_embeddings.bash
gcc -o main main.c bpe.c linear.c utils.c tensor.c attention.c -lm
./main merges.txt vocab.txt embeddings.bin corpus.txt
