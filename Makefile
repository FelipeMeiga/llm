SHELL      := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build
BIN_DIR   := bin

CC   := cc
NVCC := nvcc
MKDIR_P := mkdir -p

CPPFLAGS  := -Wall -Wextra -pedantic -I$(INC_DIR)
CFLAGS    := -std=c11 -O2 -Werror=vla
NVCCFLAGS := -std=c++17 -I$(INC_DIR)

DEBUG_CFLAGS    := -ggdb -O0 -fsanitize=address,leak,undefined
DEBUG_NVCCFLAGS := -G -g -O0 -Xcompiler="-fsanitize=address,leak,undefined"
RELEASE_FLAGS   := -O2 -DNDEBUG

.PHONY: all debug release clean

EXES := iris_test attention_test linear_test

iris_test_SRCS      := iris_test.cu utils.c tensor.cu attention.cu linear.cu
attention_test_SRCS := attention_test.cu utils.c tensor.cu attention.cu linear.cu
linear_test_SRCS    := linear_test.cu utils.c tensor.cu attention.cu linear.cu

iris_test_OBJS      := \
	$(patsubst %.c,$(BUILD_DIR)/iris_test/%.c.o,$(filter %.c,$(iris_test_SRCS))) \
	$(patsubst %.cu,$(BUILD_DIR)/iris_test/%.cu.o,$(filter %.cu,$(iris_test_SRCS)))

attention_test_OBJS := \
	$(patsubst %.c,$(BUILD_DIR)/attention_test/%.c.o,$(filter %.c,$(attention_test_SRCS))) \
	$(patsubst %.cu,$(BUILD_DIR)/attention_test/%.cu.o,$(filter %.cu,$(attention_test_SRCS)))

linear_test_OBJS    := \
	$(patsubst %.c,$(BUILD_DIR)/linear_test/%.c.o,$(filter %.c,$(linear_test_SRCS))) \
	$(patsubst %.cu,$(BUILD_DIR)/linear_test/%.cu.o,$(filter %.cu,$(linear_test_SRCS)))

all: $(addprefix $(BIN_DIR)/,$(EXES))

debug:
	$(eval CFLAGS    += $(DEBUG_CFLAGS))
	$(eval NVCCFLAGS += $(DEBUG_NVCCFLAGS))
	$(MAKE) all

release:
	$(eval CFLAGS    += $(RELEASE_FLAGS))
	$(eval NVCCFLAGS += $(RELEASE_FLAGS))
	$(MAKE) all

$(BUILD_DIR)/iris_test/%.c.o: $(SRC_DIR)/%.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/attention_test/%.c.o: $(SRC_DIR)/%.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/linear_test/%.c.o: $(SRC_DIR)/%.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/iris_test/%.cu.o: $(SRC_DIR)/%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/attention_test/%.cu.o: $(SRC_DIR)/%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/linear_test/%.cu.o: $(SRC_DIR)/%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/iris_test: $(iris_test_OBJS)
	$(MKDIR_P) $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BIN_DIR)/attention_test: $(attention_test_OBJS)
	$(MKDIR_P) $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BIN_DIR)/linear_test: $(linear_test_OBJS)
	$(MKDIR_P) $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)