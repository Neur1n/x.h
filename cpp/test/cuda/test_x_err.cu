#include "x.h"


int main(int argc, char** argv)
{
  x_err err;

  x_log('i', nullptr, "%s", err.msg());

  // CUDA driver API error
  cuInit(0);

  CUdevice device{0};
  cuDeviceGet(&device, 0);

  CUcontext context{nullptr};
  cuCtxCreate(&context, 0, device);

  err.set(x_err_cu, 1);
  x_log('e', nullptr, "%s", err.msg());

  cuCtxDestroy(context);

  // CUDA runtime API error
  err.set(x_err_cuda, 700);
  x_log('e', nullptr, "%s", err.msg());

  return 0;
}
