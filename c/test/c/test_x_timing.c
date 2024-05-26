#include "x.h"


int main(int argc, char** argv)
{
  char msg[128] = {0};
  size_t msz = x_count(msg) * sizeof(char);

  x_stopwatch ttl;
  x_stopwatch avg;

  x_stopwatch_init(&ttl);
  x_stopwatch_init(&avg);

  x_tic(&ttl, true);

  for (size_t i = 0; i < 5; ++i) {
    x_tic(&avg, false);
    x_sleep(1000);
    x_toc_avg(&avg, "ms", msg, &msz, 5, "Average", true);
  }

  x_toc(&ttl, "ms", true);
  x_log('p', NULL, "Total: %fms", ttl.elapsed);

  return 0;
}
