#include "x.h"


int main(int argc, char** argv)
{
  x_err err;

  // CUDA driver API error
  CUdeviceptr ptr{0};
  cuInit(0);

  CUdevice device{0};
  cuDeviceGet(&device, 0);

  CUcontext context{nullptr};
  cuCtxCreate(&context, 0, device);

  err.set(x_err_cuda, cuMemAlloc(&ptr, sizeof(double)));
  if (err) {
    x_log('e', nullptr, "cuMemAllocHost: %s", err.msg());
  }

  x_log('i', nullptr, "ptr type: %s", x_memtype_cu(ptr));

  if (ptr != 0) {
    err.set(x_err_cuda, cuMemFree(ptr));
    if (err) {
      x_log('e', nullptr, "cuMemFree: %s", err.msg());
    }
    ptr = 0;
  }

  cuCtxDestroy(context);

  // CUDA runtime API error
  double* mem{nullptr};

  err.set(x_err_cuda, cudaMalloc(&mem, sizeof(double)));
  if (err) {
    x_log('e', nullptr, "cudaMalloc: %s", err.msg());
  }

  x_log('i', nullptr, "mem type: %s", x_memtype_cuda(mem));

  if (mem != nullptr) {
    err.set(x_err_cuda, cudaFree(mem));
    if (err) {
      x_log('e', nullptr, "cudaFree: %s", err.msg());
    }
    mem = nullptr;
  }

  return 0;
}
