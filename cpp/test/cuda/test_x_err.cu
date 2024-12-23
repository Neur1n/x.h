#include "x.h"


int main(int argc, char** argv)
{
  x_err err;

  double* ptr{nullptr};

  x_log('i', nullptr, "%s", err.msg());

  // CUDA driver API error
  cuInit(0);

  CUdevice device{0};
  cuDeviceGet(&device, 0);

  CUcontext context{nullptr};
  cuCtxCreate(&context, 0, device);

  err.set(x_err_cu, CUDA_ERROR_OUT_OF_MEMORY);
  x_log('e', nullptr, "%s", err.msg());

  cuCtxDestroy(context);

  // CUDA runtime API error
  err.set(x_err_cuda, cudaErrorMemoryAllocation);
  x_log('e', nullptr, "%s", err.msg());

  x_check(x_err_cuda, cudaMalloc<double>, &ptr, sizeof(double));
  x_check(x_err_cuda, cudaFree, ptr);
  ptr = nullptr;

  auto faulty = [](double** ptr, size_t size) -> x_err {
    return x_err(x_err_cuda, cudaErrorMemoryAllocation);
  };
  x_check(x_err_cuda, faulty, &ptr, sizeof(double));

  return 0;
}
