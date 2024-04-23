#include "x.h"


int main(int argc, char** argv)
{
  x_err err;

  x_log('i', nullptr, "%s", err.msg());

  err.set(x_api_cuda, 700);
  x_log('e', nullptr, "%s", err.msg());

  return 0;
}
