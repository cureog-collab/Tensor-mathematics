CC = gcc

CFLAGS = -Wall -Wextra -I./include -O3 -MMD -MP
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj
BUILD_DIR = build

TARGET = $(BUILD_DIR)/tensor-maths

SRC = $(SRC_DIR)/main.c $(SRC_DIR)/tensor_elementwise.c $(SRC_DIR)/tensor_loss.c $(SRC_DIR)/tensor_init.c $(SRC_DIR)/tensor_linearAlg.c $(SRC_DIR)/tensor_reduction.c $(SRC_DIR)/tensor_utils.c
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC))
DEPS = $(OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BUILD_DIR)

-include $(DEPS)