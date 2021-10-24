#include "time_diff.h"

void TimeDiff_start(TimeDiff *t) {
    gettimeofday(&t->start, NULL);
}

void TimeDiff_stop(TimeDiff *t) {
    gettimeofday(&t->end, NULL);
}

long TimeDiff_usec(TimeDiff *t) {
    return (t->end.tv_sec - t->start.tv_sec) * 1000 * 1000 + (t->end.tv_usec - t->start.tv_usec);
}

double TimeDiff_msec(TimeDiff *t) {
    return 1e3 * (t->end.tv_sec - t->start.tv_sec) + 1e-3 *(t->end.tv_usec - t->start.tv_usec);
}

double TimeDiff_sec(TimeDiff *t) {
    return (t->end.tv_sec - t->start.tv_sec) + 1e-6 * (t->end.tv_usec - t->start.tv_usec);
}
