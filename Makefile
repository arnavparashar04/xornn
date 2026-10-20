CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -std=c23 -O2
SRC = src/main.c src/nn.c
TARGET = xornn

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) -lm

clean:
	rm -f $(TARGET)

.PHONY: all clean
