CC      = gcc
CFLAGS  = -O3 -march=native -ffast-math -std=c99 -Wall -fPIC
TARGET  = dust2_env.so

$(TARGET): dust2_env.c dust2_env.h
	$(CC) $(CFLAGS) -shared -o $(TARGET) dust2_env.c -lm

.PHONY: clean
clean:
	rm -f $(TARGET)
