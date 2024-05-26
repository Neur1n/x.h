#include "x.h"


// NOTE: This can also be a lambda function.
bool custom_fail(const int32_t val)
{
  return val != 99;
}


int main(int argc, char** argv)
{
  // NOTE: Then length of the buffer may not be enough and it is intended to
  // be truncated for testing purposes.
  char msg[32] = {0};
  size_t msz = x_count(msg);

  x_err err = x_ok();

  // NOTE: This will cause an exception.
  // err = x_err_set(x_err_custom, 98);

  err = x_err_set(x_err_custom, 99, custom_fail);
  x_log('d', NULL, "[1] %s", x_err_msg(msg, msz, err));
  if (x_fail(err)) {
    x_log('e', NULL, "[1] %s", x_err_msg(msg, msz, err));
  }

  err = x_err_set(x_err_custom, 100, custom_fail);
  x_log('w', NULL, "[2] %s", x_err_msg(msg, msz, err));
  if (x_fail(err)) {
    x_log('e', NULL, "[2] fail: %s", x_err_msg(msg, msz, err));
  }

  err = x_err_set(x_err_posix);
  x_log('i', NULL, "[4] %s", x_err_msg(msg, msz, err));

  errno = EINVAL;
  err = x_err_set(x_err_posix);
  x_log('f', NULL, "[5] %s", x_err_msg(msg, msz, err));

  return 0;
}
