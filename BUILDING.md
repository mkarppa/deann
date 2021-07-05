# Building instructions

## TL;DR

1. Install dependencies.
2. Set the environment variable `MKL_ROOT` to point to where MKL is installed.
3. Run `cmake`.
4. Run `make`.

## General requirements

* A sufficiently recent C++ compiler (tested: Apple CLang 12.0.0 / XCode 12.4,
CLang 8.0.0, GCC 9.3.0 / Ubuntu 20.04 LTS)
* Pybind11 (tested: 2.6.2 / Anaconda)
* CPython 3.8 or newer (tested: 3.8.10 / Anaconda)
* Intel MKL (tested: 2021.2.0 / Anaconda, 2020.2 / Anaconda)
* CMake 3.16 or newer (tested: 3.20.3 / Homebrew, 3.18.0, 3.16.3 /
  Ubuntu 20.04 LTS)
* NumPy (tested: 1.20.2 / Anaconda, 1.19.2 / Anaconda)

For most dependencies, it is recommended that they are acquired
through Anaconda. Although this is not strictly necessary, it will
probably make things simpler.

## Optional components

* Catch2 for building C++ unit tests (tested: 2.13.4 / Homebrew,
7727c15290ce2d289c1809d0eab3cceb88384ad6 / github,
de6fe184a9ac1a06895cdd1c9b437f0a0bdf14ad / github)
* pytest for runnign Python unit tests (tested: 6.2.3, 6.2.2 / Anaconda)
* Doxygen for building C++ documentation (tested: 1.9.1 /
  Homebrew, 1.8.17 / Ubuntu 20.04 LTS)
* Sphinx for building Python interface documentation (tested: 4.0.1 /
Anaconda, 4.10.1 / Anaconda)
* You will also need LaTeX if you want to see the math in Sphinx documentation
* scikit-learn for baseline `NearestNeighbors` module (tested: 0.24.2 /
Anaconda)
* faiss-cpu for practical nearest neighbors implementation (tested: 1.7.0 / Anaconda)

## Building the Python module for MacOS X 

The following instructions describe how the Python module can be
compiled on MacOS X. The instructions reproduce the build process as
tested on MacOS X Catalina 10.15.7. It is assumed that Homebrew is
available, and Anaconda has been installed in `~/opt/anaconda3` and
the environment has been set up correctly such that the libraries and
binaries can be found. Furthermore, Xcode and command line tools must
have been installed (tested: Xcode 12.4).

1. Install dependencies if not installed
```
$ brew install cmake
$ conda install mkl mkl-include pybind11 numpy
```
2. Install optional dependencies
```
$ brew install doxygen catch2
$ conda install sphinx pytest scikit-learn
$ conda install -c conda-forge faiss-cpu
```
3. Set the environment variable MKL_ROOT to point to the location
where MKL is installed
```
$ export MKL_ROOT=~/opt/anaconda3
```
4. Run CMake in a suitable directory (the example uses `build/`; we assume
that the working directory is the location of the source code)
```
$ mkdir build/
$ cd build/
$ cmake ..
```
No errors should be reported. You should see the following line among
the output:
```
   -- System: Darwin
```
If you did not install Catch2, you should see the following notice:
```
   -- Catch2 not found. Not building unit tests.
```
The notice should not appear if you installed Catch2 correctly and
want to build unit tests (completely optional).

5. Run Make
```
$ make
```
You should get a Python module library file, such as
`deann.cpython-38-darwin.so` in your directory (the exact name depends
on your system).

 Also, if you installed Catch2, you should get a binary called `KdeTests`.

6. If you built the unit tests, now would be a good time to check that
   the C++ interface works correctly by running `KdeTests` (completely
   optional).
   ```
   $ ./KdeTests
   ```
   This will take several minutes. The expected output is something
   similar to below:
   ```
   ===============================================================================
   All tests passed (1162893545 assertions in 88 test cases)
   ```
   
7. Check that you can load the Python module in Python (optional):
   ```
   $ python3
   >>> import deann
   >>> print(deann.__doc__)
   DEANN Python Bindings
   ```
If you get the same output and no errors, the module should be ready.

8. Verify that the Python unit tests pass (optional):
   ```
   $ pytest test.py
   ```
   The tests require certain optional components to be installed, such
   as scikit-learn, to pass. If all dependencies are met, there should
   be no errors. The expected output is as follows:
   ```
   ============================= test session starts ==============================
   platform darwin -- Python 3.8.10, pytest-6.2.3, py-1.10.0, pluggy-0.13.1
   rootdir: redacted
   plugins: anyio-2.2.0
   collected 14 items
    
   test.py ..............                                                   [100%]

   ======================== 14 passed in 137.58s (0:02:17) ========================
    ```

9. If you want to also build docs and have installed the required
dependencies:
```
$ make doc
```
The docs will be built under `docs/doc/`.

## Building the Python module for GNU/Linux

The following instructions describe how the Python module can be
compiled on GNU/Linux. The instructions reproduce the build process as
tested on Ubuntu LTS 20.04.1. It is assumed that Anaconda has been
installed in `~/anaconda3` and the environment has been set up
correctly such that the libraries and binaries can be found.

1. Install dependencies if not installed
```
$ sudo apt install cmake build-essential
$ conda install pybind11 mkl mkl-include
```

2. Install optional dependencies:
```
$ sudo apt install doxygen
$ conda install sphinx
$ conda install -c conda-forge faiss-cpu
```

For C++ unit tests, install catch2 from the github repository. Use a
v2 branch. Here we assume it is installed under `~/catch2`. Set the
environment variable:
```
$ export Catch2_DIR=~/catch2/
```

3. Set the environment variable MKL_ROOT to point to the location
where MKL is installed
```
$ export MKL_ROOT=~/anaconda3
```

4. Run CMake in a suitable directory (the example uses `build/`; we assume
that the working directory is the location of the source code)
```
$ mkdir build/
$ cd build/
$ cmake ..
```
No errors should be reported. You should see the following line among
the output:
```
   -- System: Linux
```
If you did not install Catch2, you should see the following notice:
```
   -- Catch2 not found. Not building unit tests.
```
The notice should not appear if you installed Catch2 correctly and
want to build unit tests (completely optional).
5. Run Make
```
$ make
```
You should get a Python module library file, such as
`deann.cpython-38-x86_64-linux-gnu.so` in your directory (the exact name depends
on your system).

 Also, if you installed Catch2, you should get a binary called `KdeTests`.
6. If you built the unit tests, now would be a good time to check that
   the C++ interface works correctly by running `KdeTests` (completely
   optional).
   ```
   $ ./KdeTests
   ```
   This will take several minutes. The expected output is something
   similar to below:
   ```
   ===============================================================================
   All tests passed (1162897895 assertions in 88 test cases)
   ```
7. Check that you can load the Python module in Python (optional):
   ```
   $ python3
   >>> import deann
   >>> print(deann.__doc__)
   DEANN Python Bindings
   ```
If you get the same output and no errors, the module should be ready.
8. Verify that the Python unit tests pass (optional):
   ```
   $ pytest test.py
   ```
   The tests require certain optional components to be installed, such
   as scikit-learn, to pass. If all dependencies are met, there should
   be no errors. The expected output is as follows:
   ```
============================= test session starts ==============================
platform linux -- Python 3.8.10, pytest-6.2.2, py-1.10.0, pluggy-0.13.1
rootdir: redacted
plugins: anyio-2.2.0
collected 14 items

test.py ..............                                                   [100%]

======================== 14 passed in 310.62s (0:05:10) ========================    
```
9. If you want to also build docs and have installed the required
dependencies:
```
$ make doc
```
The docs will be built under `doc/`.

## Installation the Python Module 

After completing the build step, the Python module can be installed using 
```
$ make install
```
