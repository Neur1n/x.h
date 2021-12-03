<!------------------------------------------------------------------ HEADER -->
<pre align="center">
  ▄▀▀▀▀▄▀▀▄ █▀▀▀▀▀▀▀█ █▀▀▀█▀▀▀█    █▀▀▀█▀▀▀█
  █    █  █ █   ▄▄▄▄█ █   █   █    █   █   █
  █  █ █  █ █       █ █   █   █    █       █
  █  █    █ █   ▀▀▀▀█ █   ▀   █    █   █   █
  ▀▄▄▀▄▄▄▄▀ █▄▄▄▄▄▄▄█ ▀▄▄▄▄▄▄▄▀ ██ █▄▄▄█▄▄▄█
</pre>


<p align="center">
  <img src="https://img.shields.io/badge/C++11-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++11">

  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-mit-green.svg?style=for-the-badge" alt="license-mit">
  </a>
</p>

<p align="center">
  Handy C++ utility collections.
</p>


<!------------------------------------------------------- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#documentation">Documentation</a>
      <ul>
        <li><a href="#dll_api">DLL_API</a></li>
        <li><a href="#nkey_x">NKEY_X</a></li>
        <li><a href="#nlogx">NLogX</a></li>
        <li><a href="#nbit">NBit</a></li>
        <li><a href="#ncode">NCode</a></li>
        <li><a href="#npi">NPi</a></li>
        <li><a href="#nsucc">NSucc</a></li>
        <li><a href="#nfail">NFail</a></li>
        <li><a href="#npathexists">NPathExists</a></li>
        <li><a href="#nstringempty">NStringEmpty</a></li>
        <li><a href="#ncodemessage_1">NCodeMessage [1/2]</a></li>
        <li><a href="#ncodemessage_2">NCodeMessage [2/2]</a></li>
        <li><a href="#npressedkey">NPressedKey</a></li>
        <li><a href="#nduration">NDuration</a></li>
        <li><a href="#nsleep">NSleep</a></li>
        <li><a href="#nfullpath">NFullPath</a></li>
        <li><a href="#narrarsize">NArraySize</a></li>
        <li><a href="#nclamp">NClamp</a></li>
        <li><a href="#nrelease">NRelease</a></li>
        <li><a href="#nreleasearray">NReleaseArray</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
  </ol>
</details>


<!--------------------------------------------------------- GETTING STARTED -->
## Getting Started
### Prerequisites
- C++ compiler: >= C++11
- CMake: >= 3.5


### Installation
This library contains only one header, `neu.h`. So you may download this single
file and put it into your project, or the whole repository via Git:

```bash
git clone https://github.com/Neur1n/neu.h.git
```


<!------------------------------------------------------------------- USAGE -->
## Usage
### Single Header
If you want to use only the single `neu.h` file, add the directory containing

`neu.h` to `INCLUDE_DIRECTORIES()` in your project's main `CMakeLists.txt`:
```cmake
INCLUDE_DIRECTORIES("path/to/neu.h")
```


### Whole Repository
If the whole repository is downloaded/included into your project, add the
following snippet to your project's main `CMakeLists.txt`:

```cmake
SET(NeuH_DIR "path/to/neu.h")              # Not the neu.h file, but the repository
FIND_PACKAGE(NeuH REQUIRED)                # Notice that the package name is not "neu.h"
INCLUDE_DIRECTORIES(${NeuH_INCLUDE_DIRS})  # Add to inclusion
SET(NeuH_LOG_LEVEL 0)                      # If not specified, 5 will be used by default
```

Notice that when using the whole repository, this library can be found with
`FIND_PACKAGE()` and additionally defined two variables to be used:

- `NeuH_INCLUDE_DIRS`: the include directories of neu.h
- `NeuH_LOG_LEVEL`: the log verbosity level of neu.h, of which the available

values are:
  - \>= 5: enables `NLogP`, `NLogF`, `NLogE`, `NLogW`, `NLogI`, `NLogD` (all log printings) 
  - 4: enables `NLogP`, `NLogF`, `NLogE`, `NLogW`, `NLogI`
  - 3: enables `NLogP`, `NLogF`, `NLogE`, `NLogW`
  - 2: enables `NLogP`, `NLogF`, `NLogE`
  - 1: enables `NLogP`, `NLogF`
  - 0: enables `NLogP`
  - < 0: disables all log printings

  where `P`, `F`, `E`, `W`, `I`, `D` are short for `PUBLIC`, `FATAL`, `ERROR`,
  `WARNING`, `INFO`, `DEBUG` respectively.


<!----------------------------------------------------------- DOCUMENTATION -->
## Documentation
### DLL_API
```cpp
#ifndef DLL_API
#if defined(_MSC_VER)
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __attribute__ ((visibility("default")))
#endif
#endif
```

A macro for declaring a function as a dynamic library interface.


### NKEY_X
```cpp
#define NKEY_ESC   0x1B
#define NKEY_A     0x41
#define NKEY_B     0x42
#define NKEY_C     0x43
#define NKEY_D     0x44
#define NKEY_Q     0x51
#if defined(_MSC_VER)
#define NKEY_LEFT  0x4B
#define NKEY_UP    0x48
#define NKEY_RIGHT 0x4D
#define NKEY_DOWN  0x50
#else
#define NKEY_LEFT  (-1)
#define NKEY_UP    (-2)
#define NKEY_RIGHT (-3)
#define NKEY_DOWN  (-4)
#endif
```

Several virtual key macros, most of which are defined as their corresponding
ASCII code. However, since the arrow keys are handled differently among OSs
(please tell me if this statement is not accurate), they are defined as
negative integers to work around. Please checkout
<a href="#npressedkey">NPressedKey</a> for their usage.


### NLogX
```cpp
NLogP(const char *format, ...)  // Public
NLogF(const char *format, ...)  // Fatal
NLogE(const char *format, ...)  // Error
NLogW(const char *format, ...)  // Warning
NLogI(const char *format, ...)  // Info
NLogD(const char *format, ...)  // Debug
```

Logging marcos, of which the formats of them are almost identical. What
differentiate them are the prefixes and the colors. Additionally, `NLogF` will
call `exit(EXIT_FAILURE)` to make the program exits.

<table class="center" style="text-align:center">
<thead>
  <tr>
    <th>Name</th>
    <th>Color</th>
    <th>Prefix</th>
    <th>Debug Build Format</th>
    <th>Release Build Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>NLogP</td>
    <td><span style="color:white">█</span></td>
    <td>P</td>
    <td rowspan="6">[prefix timestamp file &gt; function &gt; line] message</td>
    <td rowspan="6">[prefix timestamp] message</td>
  </tr>
  <tr>
    <td>NLogF</td>
    <td><span style="color:purple">█</span></td>
    <td>F</td>
  </tr>
  <tr>
    <td>NLogE</td>
    <td><span style="color:red">█</span></td>
    <td>E</td>
  </tr>
  <tr>
    <td>NLogW</td>
    <td><span style="color:yellow">█</span></td>
    <td>W</td>
  </tr>
  <tr>
    <td>NLogI</td>
    <td><span style="color:green">█</span></td>
    <td>I</td>
  </tr>
  <tr>
    <td>NLogD</td>
    <td><span style="color:blue">█</span></td>
    <td>D</td>
  </tr>
</tbody>
</table>

#### Parameters
- format: same as that in [printf](https://www.cplusplus.com/reference/cstdio/printf/).
- ... (optional): same as that in [printf](https://www.cplusplus.com/reference/cstdio/printf/).


### NBit
```cpp
NBit(int bit)
```

A macro used to generate an integer with only the n-th bit set to 1. This is
useful when one needs enumerations like `0b0001`, `0b0010`, `0b0100` to perform
the `&`, `|`, `~` operations.

#### Parameters
- bit: Specification for which bit of the integer should be set to 1.

#### Returns
- The generated integer.


### NCode
```cpp
typedef std::error_code NCode
```

An alias to `std::error_code`. Along with the `std::errc` aliases, it allows
much fewer typings when using the `std::error_code` mechanism. Additionally,
the naming may be more semantic than the
[POSIX errno](https://man7.org/linux/man-pages/man3/errno.3.html).


### NPi
```cpp
template<class T>
static constexpr T NPi = (T)3.141592653589793238462643383279502884197169399375;
```

A templated π for different precisions, e.g. `float`, `double`.


### NSucc
```cpp
bool NSucc(const NCode &code)
```

Check if the error code indicates an operation succeeded.

#### Parameters
- code: The error code returned by some function.

#### Returns
- `true` if the code indicates an operation succeeded.


### NFail
```cpp
bool NFail(const NCode &code)
```

Check if the error code indicates an operation failed.

#### Parameters
- code: The error code returned by some function.

#### Returns
- `true` if the code indicates an operation failed.


### NPathExists
```cpp
bool NPathExists(const char *path)
```

Check if a file or directory exits.

#### Parameters
- path: The path to the querying file/directory.

#### Returns
- `true` if the file/directory exists.


### NStringEmpty
```cpp
bool NStringEmpty(const char *string)
```

Check if a string is empty.

#### Parameters
- string: The string.

#### Returns
- `true` if the string is empty.


### NCodeMessage [1/2] <a name="ncodemessage_1"></a>
```cpp
const char *NCodeMessage(const int &code)
```

Get the description string of the error code. This function can be used to
retrieve the description of Windows OS's error code (as
[FormatMessage](https://docs.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-formatmessage) does)
or POSIX's `errno` (as [strerror](https://man7.org/linux/man-pages/man3/strerror.3.html)
does).

#### Parameters
- code: The error code.

#### Returns
- The description string of the error code.


### NCodeMessage [2/2] <a name="ncodemessage_2"></a>
```cpp
const char *NCodeMessage(const NCode &code)
```

Get the description string of a `std::error_code`.

#### Parameters
- code: The error code.

#### Returns
- The description string of the error code.


### NPressedKey
```cpp
int NPressedKey()
```

Get the pressed key. This function only works with console programs. One may
use it to break a infinity loop manually. For example:

```cpp
int main(int argc, char **argv)
{
  while (true)
  {
    if (NPressedKey() == NKEY_ESC)
    {
      break;  // When the 'Esc' key is pressed, the loop breaks.
    }

    NSleep(10);
  }

  return 0;
}
```

#### Returns
- The pressed key.


### NDuration
```cpp
long long NDuration(
    const std::chrono::system_clock::time_point &start,
    const std::chrono::system_clock::time_point &end,
    const std::string &unit = "ms")
```

Get the duration between two time points.

#### Parameters
- start: The start time point.
- end: The end time point.
- unit (optional): The unit of the outcome. Default is millisecond. Checkout
[std::chrono::duration_cast](https://www.cplusplus.com/reference/chrono/duration_cast/)
for more details.

#### Returns
- The duration of the two time points in specified unit.


### NSleep
```cpp
void NSleep(const unsigned long long &ms)
```

Blocks execution of the calling thread for specified milliseconds.

#### Parameters
- ms: The span of the sleep duration.


### NFullPath
```cpp
std::string NFullPath(const char *path)
```

Get the absolute path to a file.

#### Parameters
- path: The (maybe relative) path to a file.

#### Returns
- A `std::string` containing the absolute path to the file.


### NArraySize
```cpp
template<class T, size_t N>
size_t NArraySize(const T (&array)[N])
```

Get the element count of an array. This function does not work with those
arrays declared with the `new` operator.

#### Parameters
- array: The array of which the size to be queried.

#### Returns
- The element count of the array.


### NClamp
```cpp
template<class T>
const T &NClamp(const T &x, const T &low, const T &high)
```

Clamp the value between a lower bound and a upper bound.

#### Parameters
- x: The value to clamp.
- low: The lower bound.
- high: The upper bound.

#### Returns
- The clamped value.


### NRelease
```cpp
template<class T>
void NRelease(T &&pointer)
```

Release a pointer and make it point to `nullptr`.

#### Parameters
- pointer: The pointer to be release.


### NReleaseArray
```cpp
template<class T>
void NReleaseArray(T &&pointer)
```

Release a pointer pointed to an array and make it point to `nullptr`.

#### Parameters
- pointer: The pointer to be release.


<!----------------------------------------------------------------- LICENSE -->
## License
Distributed under the MIT license. See [LICENSE](LICENSE) for more information.
