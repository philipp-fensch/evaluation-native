#include "time_diff.h"

void TimeDiff_start(TimeDiff *t) {
    gettimeofday(&t->start, NULL);
}

void TimeDiff_stop(TimeDiff *t) {
    gettimeofday(&t->end, NULL);
}

double TimeDiff_usec(TimeDiff *t) {
    return (t->end.tv_sec - t->start.tv_sec) * 1e6 + (t->end.tv_usec - t->start.tv_usec);
}

double TimeDiff_msec(TimeDiff *t) {
    return 1e3 * (t->end.tv_sec - t->start.tv_sec) + 1e-3 *(t->end.tv_usec - t->start.tv_usec);
}

double TimeDiff_sec(TimeDiff *t) {
    return (t->end.tv_sec - t->start.tv_sec) + 1e-6 * (t->end.tv_usec - t->start.tv_usec);
}

double average_usec(TimeDiff *diffs, const int COUNT) {
    double time = 0.0;
    for(int i = 0; i < COUNT; i++) {
        time += TimeDiff_usec(&diffs[i]);
    }
    return time / COUNT;
}

double average_msec(TimeDiff *diffs, const int COUNT) {
    double time = 0.0;
    for(int i = 0; i < COUNT; i++) {
        time += TimeDiff_msec(&diffs[i]);
    }
    return time / COUNT;
}
double average_sec(TimeDiff *diffs, const int COUNT) {
    double time = 0.0;
    for(int i = 0; i < COUNT; i++) {
        time += TimeDiff_sec(&diffs[i]);
    }
    return time / COUNT;
}
