CC      = gcc
CFLAGS  = -O3 -march=native -ffast-math -std=c99 -Wall -fPIC
TARGET  = dust2_env.so
SRC     = dust2_env.c dust2_env.h

$(TARGET): dust2_env.c dust2_env.h
	$(CC) $(CFLAGS) -shared -o $(TARGET) dust2_env.c -lm

.PHONY: clean format lint
clean:
	rm -f $(TARGET)

format:
	clang-format -i $(SRC)

lint:
	clang-tidy dust2_env.c -- -std=c99 -I.
