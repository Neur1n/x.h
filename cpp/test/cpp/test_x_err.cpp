#include "x.h"


int main(int argc, char** argv)
{
  x_err err;

  err.set(x_api_custom, 99);
  x_log('d', nullptr, "%s", err.msg());

  err.set(x_api_custom, 99, "some error 99.");
  x_log('d', nullptr, "%s", err.msg());
  if (err) {
    x_log('d', nullptr, "fail");
  } else {
    x_log('d', nullptr, "succ");
  }

  err.set(x_api_posix);
  x_log('i', nullptr, "%s", err.msg());

  errno = EINVAL;
  err.set(x_api_posix);
  x_log('e', nullptr, "%s", err.msg());

  return 0;
}
