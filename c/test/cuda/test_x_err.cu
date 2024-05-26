#include "x.h"


int main(int argc, char *argv[])
{
  // NOTE: Then length of the buffer may not be enough and it is intended to
  // be truncated for testing purposes.
  char msg[32] = {0};
  size_t msz = x_count(msg);

  x_err err = x_ok();

  err = x_err_set(x_err_cuda, 700);
  x_log('e', NULL, "%s", x_err_msg(msg, msz, err));

  return 0;
}
