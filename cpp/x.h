/******************************************************************************
Copyright (c) 2023 Jihang Li
x.h is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.


Last update: 2024-05-26 11:29
Version: v0.7.0
******************************************************************************/
#ifndef X_H
#define X_H x_ver(0, 7, 0)


/** Table of Contents
 * Headers
 *                     IMPL_Compat
 * DECL_Gadget         IMPL_Gadget
 * DECL_x_log          IMPL_x_log
 * DECL_x_err          IMPL_x_err
 * DECL_x_cks          IMPL_x_cks
 * DECL_x_pkt
 * DECL_x_iov
 * DECL_x_skt          IMPL_x_skt
 * DECL_x_node         IMPL_x_node
 * DECL_x_lfque        IMPL_x_lfque
 * DECL_x_tlque        IMPL_x_tlque
 * DECL_x_stopwatch    IMPL_x_stopwatch
 * DECL_x_stopwatch_cu IMPL_x_cu_stopwatch_cu
 */

#define X_EMPTINESS

#ifndef X_ENABLE_CUDA
#define X_ENABLE_CUDA (0)
#endif

#ifndef X_ENABLE_SOCKET
#define X_ENABLE_SOCKET (0)
#endif

//************************************************* Version Number Generator{{{
/**
 * major: [0, 99]
 * minor: [0, 99]
 * patch: [0, 99999], to work with _MSC_FULL_VER
 */
#define x_ver(major, minor, patch) \
  (((major) % 100) * 10000000 + ((minor) % 100) * 100000 + ((patch) % 100000))
// Version Number Generator}}}

//*************************************************** Architecture Detection{{{
#if INTPTR_MAX == INT64_MAX
#define X_32BIT (0)
#define X_64BIT (1)
#elif INTPTR_MAX == INT32_MAX
#define X_32BIT (1)
#define X_64BIT (0)
#else
#error "Only 32-bit or 64-bit architecture is supported."
#endif

#if defined(__arm__) || defined(__thumb__) || defined(_M_ARM)
#define X_ARM (1)
#else
#define X_ARM (0)
#endif

#if defined(__aarch64__)
#define X_ARM64 (1)
#else
#define X_ARM64 (0)
#endif

#if defined(i386) || defined(__i386) || defined(__i386__) \
  || defined(__i486__) || defined(__i586__) || defined(__i686__) \
  || defined(_M_IX86_) || defined(_X86_)
#define X_X86 (1)
#else
#define X_X86 (0)
#endif

#if defined(__amd64) || defined(__amd64__) || defined(__x86_64) \
  || defined(__x86_64__) || defined(_M_AMD64) || defined(_M_X64)
#define X_X64 (1)
#else
#define X_X64 (0)
#endif
// Architecture Detection}}}

//******************************************************* Compiler Detection{{{
#if defined(__clang__)
#define X_CLANG x_ver(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
#define X_CLANG (0)
#endif

#if defined(__GNUC__)
#define X_GCC x_ver(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#else
#define X_GCC (0)
#endif

#if defined(_MSC_VER)
#define X_MSVC _MSC_FULL_VER
#else
#define X_MSVC (0)
#endif

#if defined(__NVCC__)
#define X_NVCC x_ver(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__)
#else
#define X_NVCC (0)
#endif
// Compiler Detection}}}

//*********************************************** Operating System Detection{{{
#if defined(__CYGWIX__)
#define X_CYGWIN x_ver(CYGWIN_VERSION_API_MAJOR, CYGWIN_VERSION_API_MINOR, 0)
#else
#define X_CYGWIN (0)
#endif

#if defined(__gnu_linux__) || defined(__linux__)
#define X_LINUX (1)
#else
#define X_LINUX (0)
#endif

#if defined(Macintosh) || defined(macintosh)
#define X_MACOS x_ver(9, 0, 0)
#elif defined(__APPLE__) && defined(__MACH__)
#define X_MACOS x_ver(10, 0, 0)
#else
#define X_MACOS (0)
#endif

#if defined(_WIN32) || defined(_WIN64) || defined(__WIN32__) \
  || defined(__TOS_WIX__) || defined(__WINDOWS__)
#define X_WINDOWS (1)
#else
#define X_WINDOWS (0)
#endif
// Operating System Detection}}}

//******************************************************* Platform Detection{{{
#if defined(__ANDROID__)
#define X_ANDROID (1)
#else
#define X_ANDROID (0)
#endif

#if defined(__MINGW32__) || defined(__MINGW64__)
#include <_mingw.h>
#define X_MINGW (1)
#else
#define X_MINGW (0)
#endif

#if defined(__MINGW32__)
#define X_MINGW32 x_ver(__MINGW32_VERSION_MAJOR, __MINGW32_VERSION_MINOR, 0)
#else
#define X_MINGW32 (0)
#endif

#if defined(__MINGW64__)
#define X_MINGW64 x_ver(__MINGW64_VERSION_MAJOR, __MINGW64_VERSION_MINOR, 0)
#else
#define X_MINGW64 (0)
#endif
// Platform Detection}}}

//****************************************************************** Headers{{{
#if X_CLANG || X_GCC
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include <cassert>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <format>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>

#if X_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#if X_WINDOWS && X_MSVC
#if X_ENABLE_SOCKET
#pragma comment(lib, "Ws2_32")
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <windows.h>
#include <conio.h>
#elif X_GCC || (!X_MINGW && X_CLANG)
#if X_ENABLE_SOCKET
#include <arpa/inet.h>
#include <sys/socket.h>
#endif

#include <cctype>
#include <climits>
#include <cstdlib>

#include <sys/ioctl.h>
#include <sys/sysinfo.h>
#include <termios.h>
#include <unistd.h>
#else
#error "Unsupported build environment."
#endif
// Headers}}}

//************************************************************* Env Specific{{{
// X_EXP, X_IMP
#if X_WINDOWS
#define X_EXP __declspec(dllexport)
#else
#define X_EXP __attribute__ ((visibility("default")))
#endif

#if X_WINDOWS
#define X_IMP __declspec(dllimport)
#else
#define X_IMP __attribute__ ((visibility("hidden")))
#endif

// X_KEY
#define X_KEY_ESC   (0x1B)
#define X_KEY_A     (0x41)
#define X_KEY_B     (0x42)
#define X_KEY_C     (0x43)
#define X_KEY_D     (0x44)
#define X_KEY_Q     (0x51)
#if X_WINDOWS
#define X_KEY_LEFT  (0x4B)
#define X_KEY_UP    (0x48)
#define X_KEY_RIGHT (0x4D)
#define X_KEY_DOWN  (0x50)
#else
#define X_KEY_LEFT  (-1)
#define X_KEY_UP    (-2)
#define X_KEY_RIGHT (-3)
#define X_KEY_DOWN  (-4)
#endif

// Misc
#if X_WINDOWS
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#define __PRETTY_FUNCTION__ __FUNCSIG__

#define X_PATH_MAX _MAX_PATH
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define X_PATH_MAX PATH_MAX

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS (64)
#endif
#endif
// Env Specific}}}

#define X_INL inline

//************************************************************** DECL_Gadget{{{
class x_err;

#define x_assert(expr, ...) do { \
  if (!(expr)) { \
    fprintf(stderr, "Assertion failed: %s\n", #expr); \
    if (strlen(#__VA_ARGS__)) { \
      fprintf(stderr, "Message: %s\n", __VA_ARGS__); \
    } \
    fprintf(stderr, "Position: %s:%lld: %s\n", \
        __FILENAME__, (long long)__LINE__, __PRETTY_FUNCTION__); \
    abort(); \
  } \
} while (false)

template<typename T>
static constexpr T x_Pi = static_cast<T>(3.141592653589793238462643383279502884197169399375);

template<typename T>
X_INL consteval T x_KiB(const T n);

template<typename T>
X_INL consteval T x_MiB(const T n);

template<typename T>
X_INL consteval T x_GiB(const T n);

template<typename T>
X_INL consteval T x_TiB(const T n);

template<typename T>
X_INL consteval T x_PiB(const T n);

template<typename T>
X_INL consteval
typename std::enable_if<std::is_integral_v<T>, T>::type x_bit(const T bit);

template<typename T, size_t N>
X_INL consteval size_t x_count(const T (&array)[N]);

template<typename T, bool array = false>
X_INL void x_delete(T*& ptr);

X_INL double x_duration(
    const char* unit, const struct timespec start, const struct timespec end);

#if X_ENABLE_CUDA
X_INL double x_duration_cu(
    const char* unit, const cudaEvent_t start, const cudaEvent_t stop);
#endif

// NOTE: To align with c/x.h;
X_INL bool x_fail(const x_err& err);

X_INL bool x_fexist(const char* file);

X_INL x_err x_fopen(FILE** stream, const char* file, const char* mode);

X_INL const char* x_fpath(char* dst, const char* src);

template<typename T>
X_INL void x_free(T* ptr);

template<typename T>
X_INL void x_free(volatile T* ptr);

#if X_ENABLE_CUDA
template<typename T>
X_INL void x_free_cu(T* ptr);

template<typename T>
X_INL void x_free_cu(volatile T* ptr);
#endif

X_INL long long x_fsize(const char* file);

template<typename T>
X_INL constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_gcd(const T m, const T n);

X_INL int x_getch();

template<typename T>
X_INL constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_lcm(const T m, const T n);

template<typename T>
X_INL x_err x_malloc(T** ptr, const size_t size);

#if X_ENABLE_CUDA
template<typename T>
X_INL x_err x_malloc_cu(T** ptr, const size_t size);
#endif

X_INL x_err x_memcpy(void* dst, const void* src, const size_t size);

#if X_ENABLE_CUDA
X_INL x_err x_memcpy_cu(void* dst, const void* src, const size_t size);
#endif

X_INL x_err x_meminfo(size_t* avail, size_t* total);

#if X_ENABLE_CUDA
X_INL x_err x_meminfo_cu(size_t* avail, size_t* total);
#endif

X_INL size_t x_ncpu();

X_INL size_t x_ngpu();

template<typename T>
X_INL constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_next_exp(const T base, const T src);

template<typename T>
X_INL constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_next_mul(const T base, const T src);

X_INL struct timespec x_now();

template<typename T>
X_INL constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_prev_exp(const T base, const T src);

template<typename T>
X_INL constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_prev_mul(const T base, const T src);

X_INL x_err x_split_path(
    const char* path,
    char* root, const size_t rsz, char* dir, const size_t dsz,
    char* file, const size_t fsz, char* ext, const size_t esz);

X_INL void x_sleep(const unsigned long ms);

X_INL x_err x_strcpy(char* dst, size_t dsz, const char* src);

X_INL bool x_strmty(const char* string);

// NOTE: To align with c/x.h;
X_INL bool x_succ(const x_err& err);

X_INL const char* x_timestamp(char* buf, const size_t bsz);
// DECL_Gadget}}}

//*************************************************************** DECL_x_log{{{
#define X_LOG_NONE   (-1)
#define X_LOG_PLAIN   (0)
#define X_LOG_FATAL   (1)
#define X_LOG_ERROR   (2)
#define X_LOG_WARNING (3)
#define X_LOG_INFO    (4)
#define X_LOG_DEBUG   (5)

#ifndef X_LOG_LEVEL
#ifdef NDEBUG
#define X_LOG_LEVEL X_LOG_INFO
#else
#define X_LOG_LEVEL X_LOG_DEBUG
#endif
#endif

#ifndef X_LOG_PREFIX_LIMIT
#ifdef NDEBUG
#define X_LOG_PREFIX_LIMIT (64)
#else
#define X_LOG_PREFIX_LIMIT (256)
#endif
#endif

#ifndef X_LOG_MSG_LIMIT
#define X_LOG_MSG_LIMIT (256)
#endif

template<char level, typename... Args>
X_INL void _x_log_impl(
    const char* filename, const char* function, const long line, FILE* file,
    const char* format, Args&&... args);

#define x_log(level, file, format, ...) do { \
  _x_log_impl<level>(__FILENAME__, __FUNCTION__, __LINE__, file, format, ##__VA_ARGS__); \
} while (false)
// DECL_x_log}}}

//*************************************************************** DECL_x_err{{{
enum
{
  x_err_custom = 0,
  x_err_posix  = 1,
  x_err_win32  = 2,
  x_err_socket = 3,
#if X_ENABLE_CUDA
  x_err_cuda   = 4,
#endif
  x_err_max,
#if X_WINDOWS
  x_err_system = x_err_win32,
#else
  x_err_system = x_err_posix,
#endif
};

class x_err
{
public:
  X_INL explicit x_err();

  X_INL explicit x_err(const int32_t cat);

  X_INL explicit x_err(const int32_t cat, const int32_t val);

  X_INL explicit x_err(const int32_t cat, const int32_t val, const char* msg);

  X_INL ~x_err();

  X_INL int32_t cat() const;

  X_INL const char* msg();

  X_INL x_err& set(const int32_t cat);

  X_INL x_err& set(
      const int32_t cat, const int32_t val, bool (*fail)(const int32_t) = nullptr);

  X_INL x_err& set(
      const int32_t cat, const int32_t val, const char* msg,
      bool (*fail)(const int32_t) = nullptr);

  X_INL int32_t val() const;

  X_INL operator bool() const;

private:
  int32_t m_cat{x_err_posix};
  int32_t m_val{0};
  std::string m_msg;
  bool (*m_fail)(const int32_t){nullptr};
};
// DECL_x_err}}}

//*************************************************************** DECL_x_cks{{{
X_INL uint32_t x_cks_crc32(
    const void* data, const size_t size, const uint32_t* prev);

X_INL uint16_t x_cks_rfc1071(const void* data, const size_t size);

X_INL uint8_t x_cks_xor(const void* data, const size_t size);
// DECL_x_cks}}}

//*************************************************************** DECL_x_pkt{{{
#ifndef X_PKT_SOF
#define X_PKT_SOF (0x55AA)
#endif

#ifndef X_PKT_INF
#define X_PKT_INF UINT64_MAX
#endif

typedef struct _x_hdr_
{
  uint16_t sof{X_PKT_SOF};  // start of frame
  uint16_t ctl{0};          // control code
  uint32_t opt{0};          // option, just use it freely
  uint64_t cnt{X_PKT_INF};  // total count of packets
  uint64_t idx{0};          // index of current packet
  uint64_t dsz{0};          // size of the data chunk
  uint64_t cks{0};          // checksum of packet
} x_hdr;

typedef struct _x_pkt_
{
  x_hdr head;
  void* body{nullptr};
} x_pkt;
// DECL_x_pkt}}}

//*************************************************************** DECL_x_iov{{{
typedef struct _x_iov_
{
  void* buf{nullptr};
  size_t len{0};
} x_iov;
// DECL_x_iov}}}

#if X_ENABLE_SOCKET
//*************************************************************** DECL_x_skt{{{
class x_skt
{
public:
  X_INL x_skt();

  X_INL ~x_skt();

  X_INL x_err init(const int type);

  X_INL x_err accept(x_skt* acceptee);

  X_INL x_err addr(char* ip, uint16_t* port);

  X_INL x_err close();

  X_INL x_err connect(const char* ip, const uint16_t port);

  X_INL x_err getopt(
      const int lvl, const int opt, void* val, socklen_t* len);

  X_INL x_err listen(const char* ip, const uint16_t port);

  X_INL x_err recv(void* buf, const size_t size, const int flags);

  X_INL x_err recvv(x_iov* iov, const size_t count, const int flags);

  X_INL x_err send(const void* buf, const size_t size, const int flags);

  X_INL x_err sendv(const x_iov* iov, const size_t count, const int flags);

  X_INL x_err setopt(
      const int lvl, const int opt, const void* val, const socklen_t len);

private:
#if X_WINDOWS
  SOCKET m_hndl{INVALID_SOCKET};
#else
  int m_hndl{-1};
#endif
  struct sockaddr m_addr{0};
};
// DECL_x_skt}}}
#endif  // X_ENABLE_SOCKET

//********************************************************* DECL_x_stopwatch{{{
typedef struct _x_swstats_
{
  _x_swstats_()
  {
    this->reset();
  }

  void reset()
  {
    this->ready = false;
    this->cyc = 0;
    this->sum = 0.0;
    this->avg = 0.0;
    this->max.idx = 0;
    this->max.val = DBL_MIN;
    this->min.idx = 0;
    this->min.val = DBL_MAX;
  }

  bool ready;
  size_t cyc;
  double sum;
  double avg;
  struct
  {
    size_t idx;
    double val;
  } max, min;
} x_swstats;

class x_stopwatch
{
public:
  X_INL x_stopwatch();

  X_INL ~x_stopwatch();

  X_INL double elapsed() const;

  X_INL void reset();

  X_INL void tic(const bool echo = false);

  X_INL void toc(const char* unit, const bool echo = false);

  X_INL x_swstats toc(
      const char* unit, const size_t cycle, const char* title = "",
      const bool echo = false);

private:
  struct timespec m_start{0};
  x_swstats m_stats;
  double m_elapsed{0.0};
};
// DECL_x_stopwatch}}}

//****************************************************** DECL_x_stopwatch_cu{{{
#if X_ENABLE_CUDA
class x_stopwatch_cu
{
public:
  X_INL x_stopwatch_cu();

  X_INL ~x_stopwatch_cu();

  X_INL double elapsed() const;

  X_INL void reset();

  X_INL void tic(const bool echo = false);

  X_INL void toc(const char* unit, const bool echo = false);

  X_INL x_swstats toc(
      const char* unit, const size_t cycle, const char* title = "",
      const bool echo = false);

private:
  cudaEvent_t m_start{nullptr};
  cudaEvent_t m_stop{nullptr};
  double m_elapsed{0.0};
  x_swstats m_stats;
};
#endif
// DECL_x_stopwatch_cu}}}

//************************************************************** IMPL_Compat{{{
#if !X_WINDOWS
X_INL int _kbhit()
{
  static bool initialized{false};
  if (!initialized) {
    struct termios settings{0};
    tcgetattr(STDIN_FILENO, &settings);
    settings.c_lflag &= ~ICANON;
    tcsetattr(STDIN_FILENO, TCSANOW, &settings);
    setbuf(stdin, nullptr);
    initialized = true;
  }

  int byte{0};
  ioctl(STDIN_FILENO, FIONREAD, &bytes);
  return bytes;
}
#endif
// IMPL_Compat}}}

//************************************************************** IMPL_Gadget{{{
template<typename T>
consteval T x_KiB(const T n)
{
  return n * static_cast<T>(1024);
}

template<typename T>
consteval T x_MiB(const T n)
{
  return n * static_cast<T>(1048576);
}

template<typename T>
consteval T x_GiB(const T n)
{
  return n * static_cast<T>(1073741824);
}

template<typename T>
consteval T x_TiB(const T n)
{
  return n * static_cast<T>(1099511627776);
}

template<typename T>
consteval T x_PiB(const T n)
{
  return n * static_cast<T>(1125899906842620);
}

template<typename T>
consteval typename std::enable_if<std::is_integral_v<T>, T>::type
x_bit(const T bit)
{
  return static_cast<T>(1) << bit;
}

template<typename T, size_t N>
consteval size_t x_count(const T (&array)[N])
{
  return N;
}

template<typename T, bool array>
void x_delete(T*& ptr)
{
  if (ptr != nullptr) {
    if constexpr (array) {
      delete[] ptr;
    } else {
      delete ptr;
    }
    ptr = nullptr;
  }
}

double x_duration(
    const char* unit, const struct timespec start, const struct timespec stop)
{
  double diff{static_cast<double>(
      (stop.tv_sec - start.tv_sec) * 1000000000 + stop.tv_nsec - start.tv_nsec)};

  if (strcmp(unit, "h") == 0) {
    return diff / 3600000000000.0;
  } else if (strcmp(unit, "m") == 0) {
    return diff / 60000000000.0;
  } else if (strcmp(unit, "s") == 0) {
    return diff / 1000000000.0;
  } else if (strcmp(unit, "ms") == 0) {
    return diff / 1000000.0;
  } else if (strcmp(unit, "us") == 0) {
    return diff / 1000.0;
  } else { // if (strcmp(unit, "ns") == 0)
    return diff;
  }
}

#if X_ENABLE_CUDA
double x_duration_cu(
    const char* unit, const cudaEvent_t start, const cudaEvent_t stop)
{
  cudaError_t cerr = cudaEventSynchronize(stop);
  if (cerr != cudaSuccess) {
    fprintf(stderr, "cudaEventSynchronize: %s\n", cudaGetErrorString(cerr));
    return -1.0;
  }

  float ms{0.0f};
  cerr = cudaEventElapsedTime(&ms, start, stop);
  if (cerr != cudaSuccess) {
    fprintf(stderr, "cudaEventElapsedTime: %s\n", cudaGetErrorString(cerr));
    return -1.0;
  }

  if (strcmp(unit, "h") == 0) {
    return static_cast<double>(ms) / 3600000.0;
  } else if (strcmp(unit, "m") == 0) {
    return static_cast<double>(ms) / 60000.0;
  } else if (strcmp(unit, "s") == 0) {
    return static_cast<double>(ms) / 1000.0;
  } else if (strcmp(unit, "ms") == 0) {
    return static_cast<double>(ms);
  } else if (strcmp(unit, "us") == 0) {
    return static_cast<double>(ms) * 1000.0;
  } else { // if (strcmp(unit, "ns") == 0)
    return static_cast<double>(ms) * 1000000;
  }
}
#endif

bool x_fail(const x_err& err)
{
  return err;
}

bool x_fexist(const char* file)
{
  int ierr{0};

#if X_WINDOWS
  struct _stat64 s{0};
  ierr = _stat64(file, &s);
#else
  struct stat s{0};
  ierr = stat(file, &s);
#endif

  return ierr == 0;
}

x_err x_fopen(FILE** stream, const char* file, const char* mode)
{
#if X_WINDOWS
  errno_t ierr = fopen_s(stream, file, mode);
  if (ierr != 0) {
    return x_err(x_err_posix, ierr);
  }
#else
  *stream = fopen(file, mode);
  if (*stream == nullptr) {
    return x_err(x_err_posix);
  }
#endif

  return x_err();
}

const char* x_fpath(char* dst, const char* src)
{
#if X_WINDOWS
  return dst != nullptr ? _fullpath(dst, src, X_PATH_MAX) : nullptr;
#else
  return dst != nullptr ? realpath(src, dst) : nullptr;
#endif
}

template<typename T>
void x_free(T* ptr)
{
  if (ptr != nullptr) {
    free(ptr);
    ptr = nullptr;
  }
}

template<typename T>
void x_free(volatile T* ptr)
{
  x_free<T>(const_cast<T*>(ptr));
}

#if X_ENABLE_CUDA
template<typename T>
void x_free_cu(T* ptr)
{
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

template<typename T>
void x_free_cu(volatile T* ptr)
{
  x_free_cu<T>(const_cast<T*>(ptr));
}
#endif

long long x_fsize(const char* file)
{
  int ierr{0};

#if X_WINDOWS
  struct _stat64 s{0};
  ierr = _stat64(file, &s);
#else
  struct stat s{0};
  ierr = stat(file, &s);
#endif

  return ierr == 0 ? s.st_size : -1;
}

template<typename T>
constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_gcd(const T m, const T n)
{
  return n == 0 ? m : x_gcd(n, m % n);
}

int x_getch()
{
#if X_WINDOWS
  return _kbhit() ? toupper(_getch()) : 0;
#else
  int key{0};

  int bytes_waiting{_kbhit()};
  if (bytes_waiting <= 0) {
    return 0;
  }

  struct termios old_settings{0};
  struct termios new_settings{0};
  union {
    int in;
    char ch[4];
  } buf{0};
  int ierr{0};
  ssize_t bytes_read{0};

  ierr = tcgetattr(0, &old_settings);
  if (ierr != 0) {
    return 0;
  }

  new_settings = old_settings;
  new_settings.c_lflag &= ~ICANON;
  new_settings.c_lflag &= ~ECHO;

  ierr = tcsetattr(0, TCSANOW, &new_settings);
  if (ierr != 0) {
    tcsetattr(0, TCSANOW, &old_settings);
    return 0;
  }

  bytes_read = read(STDIN_FILENO, &buf.in, bytes_waiting);
  if (bytes_read <= 0) {
    tcsetattr(0, TCSANOW, &old_settings);
    return 0;
  } else if (bytes_read >= 2) {
    if (buf.ch[0] == 0x1B && buf.ch[1] == 0x5B) {
      if (bytes_read == 2) {
        key = X_KEY_ESC;
      } else {
        switch (buf.ch[2]) {
          case X_KEY_A:
            key = X_KEY_UP;
            break;
          case X_KEY_B:
            key = X_KEY_DOWN;
            break;
          case X_KEY_C:
            key = X_KEY_RIGHT;
            break;
          case X_KEY_D:
            key = X_KEY_LEFT;
            break;
        }
      }
    } else {
      key = buf.ch[0];
    }
  } else {
    key = buf.ch[0];
  }

  tcsetattr(0, TCSADRAIN, &old_settings);

  return isalpha(key) ? toupper(key) : key;
#endif
}

template<typename T>
constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_lcm(const T m, const T n)
{
  if (m == 0 || n == 0) {
    return 0;
  }

  return m / x_gcd(m, n) * n;
}

template<typename T>
x_err x_malloc(T** ptr, const size_t size)
{
  if (*ptr != nullptr) {
    return x_err(x_err_posix, EINVAL);
  }

  *ptr = static_cast<T*>(malloc(size));
  if (*ptr == nullptr) {
    return x_err(x_err_posix);
  }

  return x_err();
}

#if X_ENABLE_CUDA
template<typename T>
X_INL x_err x_malloc_cu(T** ptr, const size_t size)
{
  if (*ptr != nullptr) {
    return x_err(x_err_posix, EINVAL);
  }

  cudaError_t cerr = cudaMalloc(ptr, size);
  if (cerr != cudaSuccess) {
    return x_err(x_err_cuda, cerr);
  }

  return x_err();
}
#endif

x_err x_memcpy(void* dst, const void* src, const size_t size)
{
  if (dst == nullptr || src == nullptr) {
    return x_err(x_err_posix, EINVAL);
  }

  if (size != 0) {
    memcpy(dst, src, size);
  }

  return x_err();
}

#if X_ENABLE_CUDA
x_err x_memcpy_cu(void* dst, const void* src, const size_t size)
{
  if (dst == nullptr || src == nullptr) {
    return x_err(x_err_posix, EINVAL);
  }

  if (size != 0) {
    cudaError_t cerr = cudaMemcpy(dst, src, size, cudaMemcpyDefault);
    if (cerr != cudaSuccess) {
      return x_err(x_err_cuda, cerr);
    }
  }

  return x_err();
}
#endif

x_err x_meminfo(size_t* avail, size_t* total)
{
  if (avail == nullptr && total == nullptr) {
    return x_err(x_err_posix, EINVAL);
  }

#if X_WINDOWS
  MEMORYSTATUSEX status{0};
  status.dwLength = sizeof(status);

  if (!GlobalMemoryStatusEx(&status)) {
    return x_err(x_err_win32);
  }

  if (avail != nullptr) {
    *avail = static_cast<size_t>(status.ullAvailPhys);
  }
  if (total != nullptr) {
    *total = static_cast<size_t>(status.ullTotalPhys);
  }
#else
  struct sysinfo info{0};
  if (sysinfo(&info) != 0) {
    return x_err(x_err_posix);
  }

  if (avail != nullptr) {
    *avail = static_cast<size_t>(info.freeram);
  }
  if (total != nullptr) {
    *total = static_cast<size_t>(info.totalram);
  }
#endif

  return x_err();
}

#if X_ENABLE_CUDA
x_err x_meminfo_cu(size_t* avail, size_t* total)
{
  if (avail == nullptr && total == nullptr) {
    return x_err(x_err_posix, EINVAL);
  }

  cudaError_t cerr = cudaMemGetInfo(avail, total);
  if (cerr != cudaSuccess) {
    return x_err(x_err_cuda, cerr);
  }

  return x_err();
}
#endif

size_t x_ncpu()
{
#if X_WINDOWS
  SYSTEM_INFO info{0};
  GetSystemInfo(&info);
  return static_cast<size_t>(info.dwNumberOfProcessors);
#else
  return static_cast<size_t>(sysconf(_SC_NPROCESSORS_ONLN));
#endif
}

size_t x_ngpu()
{
#if X_ENABLE_CUDA
  int count{0};
  cudaError_t cerr = cudaGetDeviceCount(&count);
  if (cerr != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount: %s\n", cudaGetErrorString(cerr));
    return 0;
  }

  return static_cast<size_t>(count);
#else
  return 0;
#endif
}

template<typename T>
constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_next_exp(const T base, const T src)
{
  if (src == 0) {
    return 1;
  }

  if (base == 2) {
    if ((src & (src - 1)) == 0) {
      return src;
    }

    T s{src};
    T count{0};
    while (s != 0) {
      s >>= 1;
      count += 1;
    }

    return 1 << count;
  } else {
    double exp =
      std::log(static_cast<double>(src)) / std::log(static_cast<double>(base));

    if (exp == std::round(exp)) {
      return src;
    }

    return std::pow(base, static_cast<size_t>(std::ceil(exp)));
  }
}

template<typename T>
constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_next_mul(const T base, const T src)
{
  return (src / base + 1) * base;
}

struct timespec x_now()
{
  struct timespec ts{0};

#if X_WINDOWS || __STDC_VERSION__ >= 201112L
  timespec_get(&ts, TIME_UTC);
#else
  clock_gettime(CLOCK_MONOTONIC, &ts);
#endif

  return ts;
}

template<typename T>
constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_prev_exp(const T base, const T src)
{
  if (src == 0) {
    return 0;
  }

  if (base == 2) {
    if ((src & (src - 1)) == 0) {
      return src;
    }

    T s{src};
    T count{0};
    while (s != 0) {
      s >>= 1;
      count += 1;
    }

    return 1 << (count - 1);
  } else {
    double exp =
      std::log(static_cast<double>(src)) / std::log(static_cast<double>(base));

    if (exp == std::round(exp)) {
      return src;
    }

    return std::pow(base, static_cast<size_t>(std::floor(exp)));
  }
}

template<typename T>
constexpr typename std::enable_if<std::is_integral_v<T>, T>::type
x_prev_mul(const T base, const T src)
{
  return (src / base) * base;
}

x_err x_split_path(
    const char *path,
    char *root, const size_t rsz, char *dir, const size_t dsz,
    char *file, const size_t fsz, char *ext, const size_t esz)
{
  char full[X_PATH_MAX]{0};
  x_fpath(full, path);

  if (!x_fexist(full)) {
    return x_err(x_err_posix, ENOENT);
  }

#if X_WINDOWS
  return x_err(
      x_err_posix, _splitpath_s(full, root, rsz, dir, dsz, file, fsz, ext, esz));
#else
  if (root == nullptr || rsz == 0 || dir == nullptr || dsz == 0
      || file == nullptr || fsz == 0 || ext == nullptr || esz == 0) {
    return x_err(x_err_posix, EINVAL);
  }

  if (root != nullptr) { root[0] = '\0'; }
  if (dir != nullptr) { dir[0] = '\0'; }
  if (file != nullptr) { file[0] = '\0'; }
  if (ext != nullptr) { ext[0] = '\0'; }

  size_t psz{strlen(full)};
  size_t sz{0};
  char* begin{nullptr};
  char* end{nullptr};

  // root
  begin = strchr((char*)path, '/');
  if (begin == nullptr) {
    return x_err(x_err_posix, ENOENT);
  }

  end = strchr(begin + 1, '/');
  if (end == nullptr) {
    end = full + psz;
  }

  if (root != nullptr) {
    sz = end - begin;
    if (sz >= rsz) {
      return x_err(x_err_posix, ENOBUFS);
    }

    memcpy(root, begin, sz);
    root[sz] = '\0';

    if (end == nullptr) {
      return err;
    }
  }

  // dir
  begin = strchr(end, '/');
  if (begin == nullptr) {
    return err;
  }

  end = strrchr((char*)path, '/');
  if (end <= begin) {
    return err;
  }
  if (end == nullptr) {
    end = full + psz;
  }

  if (dir != nullptr) {
    sz = end - begin;
    if (sz >= dsz) {
      return x_err(x_err_posix, ENOBUFS);
    }

    memcpy(dir, begin, sz);
    dir[sz] = '\0';

    if (end == nullptr) {
      return err;
    }
  }

  // file
  begin = end + 1;
  if ((begin - full) >= 0) {
    return err;
  }

  end = strrchr((char*)path, '.');
  if (end <= begin) {
    return err;
  }
  if (end == nullptr) {
    end = full + psz;
  }

  if (file != nullptr) {
    sz = end - begin;
    if (sz >= fsz) {
      return x_err(x_err_posix, ENOBUFS);
    }

    memcpy(file, begin, sz);
    file[sz] = '\0';
  }

  // ext
  if (ext != nullptr) {
    begin = end;
    end = full + psz;
    if (end <= begin) {
      return err;
    }

    sz = end - begin;
    memcpy(ext, begin, sz);
    ext[sz] = '\0';
  }

  return err;
#endif
}

void x_sleep(const unsigned long ms)
{
#if X_WINDOWS
  Sleep(ms);
#else
  struct timespec req{0};
  struct timespec rem{0};

  req.tv_sec = ms / 1000;
  req.tv_nsec = static_cast<long>((ms % 1000) * 1000000);

  if (nanosleep(&req, &rem) == EINTR) {
    nanosleep(&rem, nullptr);
  }
#endif
}

x_err x_strcpy(char* dst, size_t dsz, const char* src)
{
  if (dst == nullptr || dsz == 0) {
    return x_err(x_err_posix, EINVAL);
  }

  size_t cpy_sz{dsz - 1};
  size_t src_sz{strlen(src)};

  if (src_sz > 0) {
    cpy_sz = cpy_sz < src_sz ? cpy_sz : src_sz;

    x_err err = x_memcpy(dst, src, cpy_sz);
    if (err) {
      return err;
    }
  }

  dst[cpy_sz] = '\0';

  return x_err();
}

bool x_strmty(const char* string)
{
  return string == nullptr || string[0] == '\0';
}

bool x_succ(const x_err& err)
{
  return !err;
}

const char* x_timestamp(char* buf, const size_t bsz)
{
  if (buf == nullptr) {
    return "";
  }

  time_t now{time(nullptr)};

#if X_WINDOWS
  if (ctime_s(buf, bsz, &now) != 0) {
    return "";
  }
#else
  ctime_r(&now, buf);
#endif

  buf[strlen(buf) - 1] = '\0';

  return buf;
}
// IMPL_Gadget}}}

//*************************************************************** IMPL_x_log{{{
#define _X_COLOR_BLACK   "\033[90m"
#define _X_COLOR_RED     "\033[91m"
#define _X_COLOR_GREEN   "\033[92m"
#define _X_COLOR_YELLOW  "\033[93m"
#define _X_COLOR_BLUE    "\033[94m"
#define _X_COLOR_MAGENTA "\033[95m"
#define _X_COLOR_CYAN    "\033[96m"
#define _X_COLOR_WHITE   "\033[97m"
#define _X_COLOR_RESET   "\033[0m"

#define _X_LOG_COLOR_P _X_COLOR_WHITE
#define _X_LOG_COLOR_F _X_COLOR_MAGENTA
#define _X_LOG_COLOR_E _X_COLOR_RED
#define _X_LOG_COLOR_W _X_COLOR_YELLOW
#define _X_LOG_COLOR_I _X_COLOR_GREEN
#define _X_LOG_COLOR_D _X_COLOR_CYAN

template<char level>
X_INL void _x_log_prefix(
    char* buf, size_t bsz,
    const char* filename, const char* function, const long line)
{
  char timestamp[26]{0};

#ifdef NDEBUG
  snprintf(buf, bsz, "[%c %s] ", toupper(level), x_timestamp(timestamp, 26));
#else
  snprintf(
      buf, bsz, "[%c %s | %s - %s - %ld] ",
      toupper(level), x_timestamp(timestamp, 26), filename, function, line);
#endif
}

template<char level, typename... Args>
void _x_log_impl(
    const char* filename, const char* function, const long line,
    FILE* file, const char* format, Args&&... args)
{
  char color_level[8]{0};
  char color_reset[8]{0};

  if constexpr (level == 'p' || level == 'P') {
#if X_LOG_LEVEL >= X_LOG_PLAIN
    snprintf(color_level, 8, _X_LOG_COLOR_P);
#else
    return;
#endif
  } else if constexpr (level == 'f' || level == 'F') {
#if X_LOG_LEVEL >= X_LOG_FATAL
    snprintf(color_level, 8, _X_LOG_COLOR_F);
#else
    return;
#endif
  } else if constexpr (level == 'e' || level == 'E') {
#if X_LOG_LEVEL >= X_LOG_ERROR
    snprintf(color_level, 8, _X_LOG_COLOR_E);
#else
    return;
#endif
  } else if constexpr (level == 'w' || level == 'W') {
#if X_LOG_LEVEL >= X_LOG_WARNING
    snprintf(color_level, 8, _X_LOG_COLOR_W);
#else
    return;
#endif
  } else if constexpr (level == 'i' || level == 'I') {
#if X_LOG_LEVEL >= X_LOG_INFO
    snprintf(color_level, 8, _X_LOG_COLOR_I);
#else
    return;
#endif
  } else if constexpr (level == 'd' || level == 'D') {
#if X_LOG_LEVEL >= X_LOG_DEBUG
    snprintf(color_level, 8, _X_LOG_COLOR_D);
#else
    return;
#endif
  } else {
    return;
  }

  snprintf(color_reset, 8, _X_COLOR_RESET);

  char prefix[X_LOG_PREFIX_LIMIT]{0};
  _x_log_prefix<level>(prefix, X_LOG_PREFIX_LIMIT, filename, function, line);

  std::string fmsg = std::vformat(format, std::make_format_args(args...));

  // NOTE: Cover the case that there are no `{}`s in `format`.
  char msg[X_LOG_MSG_LIMIT]{0};
  snprintf(msg, X_LOG_MSG_LIMIT, fmsg.c_str(), std::forward<Args>(args)...);

  if (file == nullptr || file == stdout || file == stderr) {
    fprintf(
        file == nullptr ? stdout : file,
        "%s%s%s%s\n", color_level, prefix, msg, color_reset);
  } else {
    fprintf(file, "%s%s\n", prefix, msg);
  }
}
// IMPL_x_log}}}

//*************************************************************** IMPL_x_err{{{
x_err::x_err()
  :m_cat(x_err_posix), m_val(0)
{
}

x_err::x_err(const int32_t cat)
{
  this->set(cat);
}

x_err::x_err(const int32_t cat, const int32_t val)
{
  this->set(cat, val);
}

x_err::x_err(const int32_t cat, const int32_t val, const char* msg)
{
  this->set(cat, val, msg);
}

x_err::~x_err()
{
}

int32_t x_err::cat() const
{
  return this->m_cat;
}

const char* x_err::msg()
{
  switch (this->m_cat) {
#if X_WINDOWS
    case x_err_posix:
      if (this->m_msg.empty()) {
        this->m_msg.resize(64);
      }
      strerror_s(
          this->m_msg.data(), this->m_msg.size(), static_cast<int>(this->m_val));
      break;
    case x_err_win32:
    case x_err_socket:
      if (this->m_msg.empty()) {
        this->m_msg.resize(128);
      }
      FormatMessageA(
          FORMAT_MESSAGE_FROM_SYSTEM
          | FORMAT_MESSAGE_IGNORE_INSERTS
          | FORMAT_MESSAGE_MAX_WIDTH_MASK,
          nullptr, static_cast<DWORD>(this->m_val),
          MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
          this->m_msg.data(), this->m_msg.size(), nullptr);
      break;
#else
    case x_err_posix:
      this->m_msg = strerror(static_cast<int>(this->m_val));
      break;
#endif
#if X_ENABLE_CUDA
    case x_err_cuda:
      this->m_msg = cudaGetErrorString(static_cast<cudaError_t>(this->m_val));
      break;
#endif
    default:
      if (this->m_msg.empty()) {
        this->m_msg.resize(32);
        snprintf(
            this->m_msg.data(), this->m_msg.size(), "Custom error %d", this->m_val);
      }
  }

  return this->m_msg.c_str();
}

x_err& x_err::set(const int32_t cat)
{
  switch (cat) {
    case x_err_posix:
      this->m_cat = cat;
      this->m_val = static_cast<int32_t>(errno);
      break;
#if X_WINDOWS
    case x_err_win32:
      this->m_cat = cat;
      this->m_val = static_cast<int32_t>(GetLastError());
      break;
#if X_ENABLE_SOCKET
    case x_err_socket:
      this->m_cat = cat;
      this->m_val = static_cast<int32_t>(WSAGetLastError());
      break;
#endif
#endif
#if X_ENABLE_CUDA
    case x_err_cuda:
      this->m_cat = cat;
      this->m_val = static_cast<int32_t>(cudaGetLastError());
      break;
#endif
    default:
      throw std::invalid_argument(
          std::string("Unsupported error category: ") + std::to_string(cat));
  }

  this->m_msg.clear();

  return *this;
}

x_err& x_err::set(
    const int32_t cat, const int32_t val, bool (*fail)(const int32_t))
{
  if (fail == nullptr && (cat <= x_err_custom || cat >= x_err_max)) {
    throw std::invalid_argument(
        "A failure predicate is required for a custom error.");
  }

  this->m_cat = cat;
  this->m_val = val;
  this->m_msg.clear();
  this->m_fail = fail;

  return *this;
}

x_err& x_err::set(
    const int32_t cat, const int32_t val, const char* msg,
    bool (*fail)(const int32_t))
{
  if (fail == nullptr && (cat <= x_err_custom || cat >= x_err_max)) {
    throw std::invalid_argument(
        "A failure predicate is required for a custom error.");
  }

  this->m_cat = cat;
  this->m_val = val;
  this->m_msg = msg;
  this->m_fail = fail;

  return *this;
}

int32_t x_err::val() const
{
  return this->m_val;
}

x_err::operator bool() const
{
  if (this->m_fail) {
    // NOTE: Covers the x_err_custom and other customized cases.
    return this->m_fail(this->m_val);
  } else {
    switch (this->m_cat) {
#if X_WINDOWS
      case x_err_win32:
        return this->m_val != 0;
#if X_ENABLE_SOCKET
      case x_err_socket:
        return this->m_val != 0;
#endif
#endif
#if X_ENABLE_CUDA
      case x_err_cuda:
        return this->m_val != cudaSuccess;
#endif
      default:
        // NOTE: Covers the x_err_posix case.
        return this->m_val != 0;
    }
  }
}
// IMPL_x_err}}}

//*************************************************************** IMPL_x_cks{{{
uint32_t x_cks_crc32(const void* data, const size_t size, const uint32_t* prev)
{
  uint8_t* d{(uint8_t*)data};
  size_t cnt{size / sizeof(uint8_t)};
  int i{0};

  uint32_t cks{prev ? ~(*prev) : 0xFFFFFFFF};

  while (cnt--) {
    cks ^= *d++;

    for (i = 0; i < 8; ++i) {
      cks = (cks >> 1) ^ (-static_cast<int32_t>(cks & 1) & 0xEDB88320);
    }
  }

  return ~cks;
}

uint16_t x_cks_rfc1071(const void* data, const size_t size)
{
  uint16_t* d{(uint16_t*)data};
  size_t cnt{size / sizeof(uint8_t)};
  uint32_t cks{0};

  while (cnt > 1) {
    cks += *d++;
    cnt -= 2;
  }

  if (cnt > 0) {
    cks += *d;
  }

  while (cks >> 16) {
    cks = (cks & 0xFFFF) + (cks >> 16);
  }

  return static_cast<uint16_t>(~cks);
}

uint8_t x_cks_xor(const void* data, const size_t size)
{
  const uint8_t* d8{(const uint8_t*)data};
  const uint64_t* d64{(const uint64_t*)data};
  const size_t dsz{sizeof(uint64_t)};
  const size_t cnt{size / dsz};

  union {
    uint8_t u8[8];
    uint64_t u64;
  } cks{{0}};

  size_t i{0};
  for (i = 0; i < (cnt & (~0x07)); i += 8) {
    cks.u64 ^= d64[i] ^ d64[i + 1] ^ d64[i + 2] ^ d64[i + 3]
      ^ d64[i + 4] ^ d64[i + 5] ^ d64[i + 6] ^ d64[i + 7];
  }

  for (size_t j = i * dsz; j < size; ++j) {
    cks.u8[0] ^= d8[j];
  }

  cks.u8[0] ^= cks.u8[1] ^ cks.u8[2] ^ cks.u8[3]
    ^ cks.u8[4] ^ cks.u8[5] ^ cks.u8[6] ^ cks.u8[7];

  return cks.u8[0];
}
// IMPL_x_cks}}}

#if X_ENABLE_SOCKET
//*************************************************************** IMPL_x_skt{{{
x_skt::x_skt()
{
}

x_err x_skt::init(const int type)
{
#if X_WINDOWS
  WSADATA data{0};
  if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
    return x_err(x_err_socket);
  }
#endif

  if (type == SOCK_STREAM) {
    this->m_hndl = socket(AF_INET, type, IPPROTO_TCP);
  } else if (type == SOCK_DGRAM) {
    this->m_hndl = socket(AF_INET, type, IPPROTO_UDP);
  } else {
    return x_err(x_err_posix, ENOTSUP);
  }

#if X_WINDOWS
  if (this->m_hndl == INVALID_SOCKET) {
    return x_err(x_err_socket);
  }
#else
  if (this->m_hndl == -1) {
    return x_err(x_err_socket);
  }
#endif

  int val{1};
  socklen_t len{static_cast<socklen_t>(sizeof(val))};
  setsockopt(this->m_hndl, SOL_SOCKET, SO_KEEPALIVE, (char*)&val, len);
#if X_WINDOWS
  setsockopt(this->m_hndl, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (char*)&val, len);
#else
  val = 0;
  setsockopt(this->m_hndl, SOL_SOCKET, SO_REUSEADDR, (char*)&val, len);
#endif

  return x_err();
}

x_err x_skt::accept(x_skt* acceptee)
{
  if (acceptee == nullptr) {
    return x_err(x_err_posix, EINVAL);
  }

  struct sockaddr_in sin{0};
  socklen_t len{0};

#if X_WINDOWS
  SOCKET hndl = ::accept(this->m_hndl, (struct sockaddr*)&sin, &len);
  if (hndl == INVALID_SOCKET) {
#else
  int hndl = ::accept(this->m_hndl, (struct sockaddr*)&sin, &len);
  if (hndl == -1) {
#endif
    return x_err(x_err_socket);
  }

  struct sockaddr addr{0};
  memcpy(&addr, &sin, len);

  acceptee->m_addr = std::move(addr);
  acceptee->m_hndl = std::move(hndl);

  return x_err();
}

x_err x_skt::addr(char* ip, uint16_t* port)
{
  if (ip == nullptr || port == nullptr) {
    return x_err(x_err_posix, EINVAL);
  }

  struct sockaddr_in* sin{(struct sockaddr_in*)&this->m_addr};

  if (inet_ntop(AF_INET, &sin->sin_addr, ip, 16) == nullptr) {
    return x_err(x_err_socket);
  }

  *port = sin->sin_port;

  return x_err();
}

x_err x_skt::close()
{
#if X_WINDOWS
  if (closesocket(this->m_hndl) != 0) {
    return x_err(x_err_socket);
  }

  return WSACleanup() == 0 ? x_err() : x_err(x_err_socket);
#else
  return ::close(this->m_hndl) == 0 ? x_err() : x_err(x_err_socket);
#endif
}

x_err x_skt::connect(const char* ip, const uint16_t port)
{
  struct sockaddr_in sin{0};
  sin.sin_family = AF_INET;
  sin.sin_port = htons(port);
  int ierr{inet_pton(AF_INET, ip, &sin.sin_addr)};
  if (ierr == 0) {
    return x_err(x_err_posix, EFAULT);
  } else if (ierr == -1) {
    return x_err(x_err_socket);
  }

  memcpy(&this->m_addr, &sin, sizeof(struct sockaddr));

  return ::connect(this->m_hndl, &this->m_addr, sizeof(struct sockaddr_in)) == 0
    ? x_err() : x_err(x_err_socket);
}

x_err x_skt::getopt(const int lvl, const int opt, void* val, socklen_t* len)
{
  if (val == nullptr || len == nullptr) {
    return x_err(x_err_posix, EINVAL);
  }

  return getsockopt(this->m_hndl, lvl, opt, (char*)val, len) == 0
    ? x_err() : x_err(x_err_socket);
}

x_err x_skt::listen(const char* ip, const uint16_t port)
{
  struct sockaddr_in sin{0};
  sin.sin_family = AF_INET;
  sin.sin_port = htons(port);

  int ierr{inet_pton(AF_INET, ip, &sin.sin_addr)};
  if (ierr == 0) {
    return x_err(x_err_posix, EFAULT);
  } else if (ierr == -1) {
    return x_err(x_err_socket);
  }

  memcpy(&this->m_addr, &sin, sizeof(struct sockaddr));

  ierr = bind(this->m_hndl, &this->m_addr, sizeof(struct sockaddr_in));
  if (ierr == 0) {
    ierr = ::listen(this->m_hndl, SOMAXCONN);
  }

  return ierr == 0 ? x_err() : x_err(x_err_socket);
}

x_err x_skt::recv(void* buf, const size_t size, const int flags)
{
  if (buf == nullptr || size == 0) {
    return x_err(x_err_posix, EINVAL);
  }

#if X_WINDOWS
  int remain{static_cast<int>(size)};
  int bytes{0};
#else
  size_t remain{size};
  ssize_t bytes{0};
#endif
  size_t offset{0};

  while (remain > 0) {
    bytes = ::recv(this->m_hndl, static_cast<char*>(buf) + offset, remain, flags);
    if (bytes <= 0) {
      return x_err(x_err_socket);
    }

    offset += bytes;
    remain -= bytes;
  }

  return x_err();
}

x_err x_skt::recvv(x_iov* iov, const size_t count, const int flags)
{
  if (iov == nullptr || count == 0) {
    return x_err(x_err_posix, EINVAL);
  }

  size_t total{0};
  for (size_t i = 0; i < count; ++i) {
    if (iov[i].buf == nullptr || iov[i].len == 0) {
      return x_err(x_err_posix, EINVAL);
    }

    if (iov[i].len > (SIZE_MAX - total)) {
      return x_err(x_err_posix, EOVERFLOW);
    }

    total += iov[i].len;
  }

  // NOTE: _alloca/alloca may be used if all data received are rather small.
  void* buf = malloc(total);
  if (buf == nullptr) {
    return x_err(x_err_posix);
  }

  x_err err = this->recv(buf, total, flags);
  if (!err) {
    size_t offset{0};
    for (size_t i = 0; i < count; ++i) {
      memcpy(iov[i].buf, (char*)buf + offset, iov[i].len);
      offset += iov[i].len;
    }
  }

  free(buf);

  return err;
}

x_err x_skt::send(const void* buf, const size_t size, const int flags)
{
#if X_WINDOWS
  int remain{static_cast<int>(size)};
#else
  size_t remain{size};
#endif
  size_t offset{0};
  int bytes{0};

  while (remain > 0) {
    bytes = ::send(this->m_hndl, (char*)buf + offset, remain, flags);
    if (bytes <= 0) {
      return x_err(x_err_socket);
    }

    offset += bytes;
    remain -= bytes;
  }

  return x_err();
}

x_err x_skt::sendv(const x_iov* iov, const size_t count, const int flags)
{
  if (iov == nullptr || count == 0) {
    return x_err(x_err_posix, EINVAL);
  }

  size_t total{0};
  for (size_t i = 0; i < count; ++i) {
    if (iov[i].buf == nullptr || iov[i].len == 0) {
      return x_err(x_err_posix, EINVAL);
    }

    if (iov[i].len > (SIZE_MAX - total)) {
      return x_err(x_err_posix, EOVERFLOW);
    }

    total += iov[i].len;
  }

  // NOTE: _alloca/alloca may be used if all data sent are rather small.
  void* buf = malloc(total);
  if (buf == nullptr) {
    return x_err(x_err_posix);
  }

  size_t offset{0};
  for (size_t i = 0; i < count; ++i) {
    memcpy(static_cast<char*>(buf) + offset, iov[i].buf, iov[i].len);
    offset += iov[i].len;
  }

  x_err err = this->send(buf, total, flags);

  free(buf);

  return err;
}

x_err x_skt::setopt(
    const int lvl, const int opt, const void* val, const socklen_t len)
{
  if (val == nullptr) {
    return x_err(x_err_posix, EINVAL);
  }

  return setsockopt(this->m_hndl, lvl, opt, (char*)val, len) == 0
    ? x_err() : x_err(x_err_socket);
}
// IMPL_x_skt}}}
#endif  // X_ENABLE_SOCKET

//********************************************************* IMPL_x_stopwatch{{{
x_stopwatch::x_stopwatch()
{
}

x_stopwatch::~x_stopwatch()
{
}

double x_stopwatch::elapsed() const
{
  return this->m_elapsed;
}

void x_stopwatch::reset()
{
  this->m_stats.reset();
  this->m_elapsed = 0.0;
}

void x_stopwatch::tic(const bool echo)
{
  this->m_start = x_now();

  if (echo) {
    char ts[26]{0};
    printf("Stopwatch starts at: %s.\n", x_timestamp(ts, x_count(ts)));
  }
}

void x_stopwatch::toc(const char* unit, const bool echo)
{
  this->m_elapsed = x_duration(unit, this->m_start, x_now());

  if (echo) {
    char ts[26]{0};
    printf("Stopwatch stops at: %s (%f%s elapsed).\n",
        x_timestamp(ts, x_count(ts)), this->m_elapsed, unit);
  }
}

x_swstats x_stopwatch::toc(
    const char* unit, const size_t cycle, const char* title, const bool echo)
{
  if (cycle == 0) {
    this->m_stats.reset();
    return this->m_stats;
  }

  this->toc(unit, false);

  if (this->m_elapsed > this->m_stats.max.val) {
    this->m_stats.max.idx = this->m_stats.cyc;
    this->m_stats.max.val = this->m_elapsed;
  }
  if (this->m_elapsed < this->m_stats.min.val) {
    this->m_stats.min.idx = this->m_stats.cyc;
    this->m_stats.min.val = this->m_elapsed;
  }

  this->m_stats.sum += this->m_elapsed;
  this->m_stats.cyc += 1;
  this->m_stats.avg = this->m_stats.sum / this->m_stats.cyc;

  if (this->m_stats.cyc % cycle == 0) {
    this->m_stats.ready = true;

    const char* t = x_strmty(title) ? "STATS" : title;

    std::string msg(128, '\0');

    msg = std::string("[") + t + std::string("] ")
      + std::to_string(this->m_stats.sum) + unit + " in "
      + std::to_string(this->m_stats.cyc) + " cycles - avg: "
      + std::to_string(this->m_stats.avg) + unit + ", min("
      + std::to_string(this->m_stats.min.idx) + "): "
      + std::to_string(this->m_stats.min.val) + unit + ", max("
      + std::to_string(this->m_stats.max.idx) + "): "
      + std::to_string(this->m_stats.max.val) + unit;

    if (echo) {
      printf("%s\n", msg.c_str());
    }
  }

  return this->m_stats;
}
// IMPL_x_stopwatch}}}

#if X_ENABLE_CUDA
//****************************************************** IMPL_x_stopwatch_cu{{{
x_stopwatch_cu::x_stopwatch_cu()
{
  cudaError_t cerr = cudaEventCreate(&this->m_start);
  if (cerr != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaEventCreate: ") + cudaGetErrorString(cerr));
  }

  cerr = cudaEventCreate(&this->m_stop);
  if (cerr != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaEventCreate: ") + cudaGetErrorString(cerr));
  }
}

x_stopwatch_cu::~x_stopwatch_cu()
{
  cudaEventDestroy(this->m_start);
  cudaEventDestroy(this->m_stop);
}

double x_stopwatch_cu::elapsed() const
{
  return this->m_elapsed;
}

void x_stopwatch_cu::reset()
{
  this->m_stats.reset();
  this->m_elapsed = 0.0;
}

void x_stopwatch_cu::tic(const bool echo)
{
  cudaError_t cerr = cudaEventRecord(this->m_start);
  if (cerr != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaEventRecord: ") + cudaGetErrorString(cerr));
  }

  if (echo) {
    char ts[26]{0};
    printf("Stopwatch starts at: %s.\n", x_timestamp(ts, x_count(ts)));
  }
}

void x_stopwatch_cu::toc(const char* unit, const bool echo)
{
  cudaError_t cerr = cudaEventRecord(this->m_stop);
  if (cerr != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaEventRecord: ") + cudaGetErrorString(cerr));
  }

  cerr = cudaEventSynchronize(this->m_stop);
  if (cerr != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaEventSynchronize: ") + cudaGetErrorString(cerr));
  }

  float elapsed{0.0f};
  cerr = cudaEventElapsedTime(&elapsed, this->m_start, this->m_stop);
  if (cerr != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaEventElapsedTime: ") + cudaGetErrorString(cerr));
  }

  if (strcmp(unit, "h") == 0) {
    elapsed /= 3600000.0f;
  } else if (strcmp(unit, "m") == 0) {
    elapsed /= 60000.0f;
  } else if (strcmp(unit, "s") == 0) {
    elapsed /= 1000.0f;
  } else if (strcmp(unit, "us") == 0) {
    elapsed *= 1000.0f;
  } else if (strcmp(unit, "ns") == 0) {
    elapsed *= 1000000.0f;
  }
  this->m_elapsed = static_cast<double>(elapsed);

  if (echo) {
    char ts[26]{0};
    printf("Stopwatch stops at: %s (%f%s elapsed).\n",
        x_timestamp(ts, x_count(ts)), this->m_elapsed, unit);
  }
}

x_swstats x_stopwatch_cu::toc(
    const char* unit, const size_t cycle, const char* title, const bool echo)
{
  if (cycle == 0) {
    this->m_stats.reset();
    return this->m_stats;
  }

  this->toc(unit, false);

  if (this->m_elapsed > this->m_stats.max.val) {
    this->m_stats.max.idx = this->m_stats.cyc;
    this->m_stats.max.val = this->m_elapsed;
  }
  if (this->m_elapsed < this->m_stats.min.val) {
    this->m_stats.min.idx = this->m_stats.cyc;
    this->m_stats.min.val = this->m_elapsed;
  }

  this->m_stats.sum += this->m_elapsed;
  this->m_stats.cyc += 1;
  this->m_stats.avg = this->m_stats.sum / this->m_stats.cyc;

  if (this->m_stats.cyc % cycle == 0) {
    this->m_stats.ready = true;

    if (echo) {
      const char* t = x_strmty(title) ? "STATS" : title;

      std::string msg(128, '\0');

      msg = std::string("[") + t + std::string("] ")
        + std::to_string(this->m_stats.sum) + unit + " in "
        + std::to_string(this->m_stats.cyc) + " cycles - avg: "
        + std::to_string(this->m_stats.avg) + unit + ", min("
        + std::to_string(this->m_stats.min.idx) + "): "
        + std::to_string(this->m_stats.min.val) + unit + ", max("
        + std::to_string(this->m_stats.max.idx) + "): "
        + std::to_string(this->m_stats.max.val) + unit;

      printf("%s\n", msg.c_str());
    }
  }

  return this->m_stats;
}
// IMPL_x_stopwatch_cu}}}
#endif


#endif  // X_H
