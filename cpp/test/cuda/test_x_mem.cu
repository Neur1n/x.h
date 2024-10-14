#include "x.h"


int main(int argc, char** argv)
{
  x_err err;

  double* ptr{nullptr};

  err = x_malloc_cuda(&ptr, sizeof(double));
  if (err) {
    x_log('e', nullptr, "x_malloc: %s", err.msg());
    return EXIT_FAILURE;
  }

  cudaPointerAttributes attr;
  cudaError_t cerr = cudaPointerGetAttributes(&attr, ptr);
  if (cerr != cudaSuccess) {
    err.set(x_err_cuda, cerr);
    x_log('e', nullptr, "cudaPointerGetAttributes: %s", err.msg());
    return EXIT_FAILURE;
  }

  x_log('i', nullptr, "Pointer type: %d", static_cast<int>(attr.type));

  x_free_cuda(ptr);

  return 0;
}
