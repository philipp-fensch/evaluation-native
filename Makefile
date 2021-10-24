CC = gcc
LIBS = cudart
SRC = main.c time_diff.c

.PHONY: clean

all: eval_native

eval_native: $(SRC)
	$(CC) -o eval_native $^ -I/usr/local/cuda/include -l$(LIBS) -O3

clean:
	@echo clean eval_native
	rm -f eval_native