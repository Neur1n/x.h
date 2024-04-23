#include "x.h"


int main(int argc, char** argv)
{
  x_timing<x_api_gpu> ttl;
  x_timing<x_api_gpu> avg;

  ttl.tic(true);

  for (size_t i = 0; i < 5; ++i) {
    avg.tic();
    x_sleep(1000);
    avg.toc("ms", 5, "Average", true);
  }

  ttl.toc("ms", true);
  x_log('p', nullptr, "Total: %fms", ttl.elapsed());

  return 0;
}
