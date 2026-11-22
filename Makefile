# NeuralNetwork — Build System (GNU Make)
# Usage:
#   make                Build library, apps, and tests (debug mode)
#   make MODE=release   Build with optimizations (-O3 -march=native)
#   make test           Build and execute all unit tests
#   make clean          Remove all build artifacts

CC       := gcc
CXX      := g++
AR       := ar
MODE     ?= debug
SANITIZE ?= 0

ifeq ($(MODE),release)
    OPT_FLAGS := -O3 -march=native -DNDEBUG
else ifeq ($(MODE),debug)
    OPT_FLAGS := -O0 -g3
else
    $(error Unknown MODE '$(MODE)'. Use 'debug' or 'release')
endif

COMMON_CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Wshadow -MMD -MP -Iinclude -isystem third_party/cjson
CXXFLAGS        := $(COMMON_CXXFLAGS) $(OPT_FLAGS)
CJSON_CFLAGS    := -std=c99 -MMD -MP $(OPT_FLAGS)
LDFLAGS         := -pthread

ifeq ($(SANITIZE),1)
    CXXFLAGS     += -fsanitize=address,undefined
    CJSON_CFLAGS += -fsanitize=address,undefined
    LDFLAGS      += -fsanitize=address,undefined
endif

BUILD_DIR := build/$(MODE)
OBJ_DIR   := $(BUILD_DIR)/obj
TEST_DIR  := $(BUILD_DIR)/tests
APP_DIR   := $(BUILD_DIR)/apps

CJSON_SRC := third_party/cjson/cJSON.c
CJSON_OBJ := $(OBJ_DIR)/third_party/cjson/cJSON.o

LIB_SRCS   := $(wildcard src/*.cpp)
LIB_OBJS   := $(patsubst src/%.cpp,$(OBJ_DIR)/src/%.o,$(LIB_SRCS)) $(CJSON_OBJ)
LIB_TARGET := $(BUILD_DIR)/libnn.a

TEST_SRCS := $(wildcard tests/test_*.cpp)
TEST_OBJS := $(patsubst tests/%.cpp,$(OBJ_DIR)/tests/%.o,$(TEST_SRCS))
TEST_BINS := $(patsubst tests/%.cpp,$(TEST_DIR)/%,$(TEST_SRCS))

APP_SRCS  := $(wildcard apps/*.cpp)
APP_OBJS  := $(patsubst apps/%.cpp,$(OBJ_DIR)/apps/%.o,$(APP_SRCS))
APP_BINS  := $(patsubst apps/%.cpp,$(APP_DIR)/%,$(APP_SRCS))

DEPS := $(LIB_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(APP_OBJS:.o=.d)

.SECONDARY: $(TEST_OBJS) $(APP_OBJS) $(CJSON_OBJ)

.PHONY: all lib apps tests test clean help

all: lib apps tests

lib: $(LIB_TARGET)

apps: $(APP_BINS)

tests: $(TEST_BINS)

$(LIB_TARGET): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(CJSON_OBJ): $(CJSON_SRC)
	@mkdir -p $(dir $@)
	$(CC) $(CJSON_CFLAGS) -c $< -o $@

$(OBJ_DIR)/src/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/tests/%.o: tests/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/apps/%.o: apps/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TEST_DIR)/%: $(OBJ_DIR)/tests/%.o $(LIB_TARGET)
	@mkdir -p $(dir $@)
	$(CXX) $^ $(LDFLAGS) -o $@

$(APP_DIR)/%: $(OBJ_DIR)/apps/%.o $(LIB_TARGET)
	@mkdir -p $(dir $@)
	$(CXX) $^ $(LDFLAGS) -o $@

test: tests
	@set -e; \
	for bin in $(TEST_BINS); do \
		echo "=== Running $$bin ==="; \
		$$bin; \
	done

clean:
	rm -rf build

help:
	@echo "Available targets:"
	@echo "  make [MODE=debug|release] [SANITIZE=0|1]  Build lib, apps, and tests"
	@echo "  make lib                                  Build static library libnn.a"
	@echo "  make apps                                 Build executables in apps/"
	@echo "  make tests                                Build unit tests"
	@echo "  make test                                 Build and run all unit tests"
	@echo "  make clean                                Remove build/ directory"

-include $(DEPS)
