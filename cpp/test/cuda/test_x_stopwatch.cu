#include "x.h"


int main(int argc, char** argv)
{
  x_stopwatch_stats stats;

  // CUDA driver API event
  cuInit(0);

  CUdevice device{0};
  cudaGetDevice(&device);

  CUcontext context{nullptr};
  cuCtxCreate(&context, 0, device);

  x_stopwatch_cu ttl_cu;
  x_stopwatch_cu avg_cu;

  ttl_cu.tic();
  x_log('i', nullptr, "[ttl] x_stopwatch_cu starts.");

  for (size_t i = 0; i < 5; ++i) {
    avg_cu.tic();
    x_sleep(1000);
    avg_cu.toc(stats, "ms", 3);
  }
  if (stats.ready) {
    stats.echo();
  }

  stats.reset();

  for (size_t i = 0; i < 5; ++i) {
    avg_cu.tic();
    x_sleep(1000);
    avg_cu.toc(stats, "ms", 5);
  }
  if (stats.ready) {
    stats.echo();
  }

  ttl_cu.toc("ms");
  x_log('i', nullptr, "[ttl] x_stopwatch_cu stops.", ttl_cu.elapsed());
  x_log('p', nullptr, "[ttl] Total: %fms", ttl_cu.elapsed());

  cuCtxDestroy(context);

  printf("\n");

  // CUDA runtime API event
  x_stopwatch_cuda ttl_cuda;
  x_stopwatch_cuda avg_cuda;

  ttl_cuda.tic();
  x_log('i', nullptr, "[ttl] x_stopwatch_cuda starts.");

  for (size_t i = 0; i < 5; ++i) {
    avg_cuda.tic();
    x_sleep(1000);
    avg_cuda.toc(stats, "ms", 3);
  }
  if (stats.ready) {
    stats.echo();
  }

  stats.reset();

  for (size_t i = 0; i < 5; ++i) {
    avg_cuda.tic();
    x_sleep(1000);
    avg_cuda.toc(stats, "ms", 5);
  }
  if (stats.ready) {
    stats.echo();
  }

  ttl_cuda.toc("ms");
  x_log('i', nullptr, "[ttl] x_stopwatch_cuda stops.", ttl_cuda.elapsed());
  x_log('p', nullptr, "[ttl] Total: %fms", ttl_cuda.elapsed());

  return 0;
}
