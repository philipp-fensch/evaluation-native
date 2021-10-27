#ifndef TIMEDIFF_H
#define TIMEDIFF_H

#include <sys/time.h>
#include <stddef.h>

typedef struct {
    struct timeval start;
    struct timeval end;
} TimeDiff;

// Set time
void TimeDiff_start(TimeDiff *t);
void TimeDiff_stop(TimeDiff *t);

// Retrieve difference
double TimeDiff_usec(TimeDiff *t);
double TimeDiff_msec(TimeDiff *t);
double TimeDiff_sec(TimeDiff *t);

double average_usec(TimeDiff *diffs, const int COUNT);
double average_msec(TimeDiff *diffs, const int COUNT); 
double average_sec(TimeDiff *diffs, const int COUNT); 

#endif // TIMEDIFF_H
