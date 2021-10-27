#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "matrix.h"

double *read_matrix_from_file(const int DIM, const char *file_name) {
    return read_vector_from_file(DIM * DIM, file_name);
}

double *read_vector_from_file(const int DIM, const char *file_name) {
    int fd = open(file_name, O_RDONLY);
    int len = lseek(fd, 0, SEEK_END);
    char *data = (char*)mmap(0, len, PROT_READ, MAP_PRIVATE, fd, 0);

    double *matrix = (double*) malloc(DIM * sizeof(double));
    for(int i = 0; i < DIM; i++) {
        matrix[i] = strtod(data, &data);
    }
    return matrix;
}
