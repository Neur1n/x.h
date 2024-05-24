#include "x.h"


int main(int argc, char** argv)
{
  x_stopwatch ttl;
  x_stopwatch avg;

  ttl.tic(true);

  for (size_t i = 0; i < 5; ++i) {
    avg.tic();
    x_sleep(1000);
    avg.toc("ms", 5, "Average", true);
  }

  ttl.toc("ms", false);
  x_log('p', nullptr, "Total: %fms", ttl.elapsed());

  return 0;
}
