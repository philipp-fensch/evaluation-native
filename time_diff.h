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
long TimeDiff_usec(TimeDiff *t);
double TimeDiff_msec(TimeDiff *t);
double TimeDiff_sec(TimeDiff *t);

#endif // TIMEDIFF_H
