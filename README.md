# Image Convolution - SIMD & Parallel Optimization

High-performance C++ implementation of image convolution using AVX intrinsics and OpenMP parallelization.

## Features

**Implementation Variants**
- Baseline sequential convolution
- OpenMP multi-threaded processing
- SIMD optimization with AVX (256-bit registers)
- Combined SIMD + OpenMP (best performance)

**Optimizations**
- AVX intrinsics: `_mm256_mul_pd`, `_mm256_add_pd` for vectorized operations
- OpenMP parallelization with static scheduling and reduction clauses
- Image padding for edge handling

**Performance Testing**
- Warm-up and multi-iteration measurement
- Statistical analysis (mean time, variance)
- Timing with `omp_get_wtime()`

## Tech Stack

- C++ with OpenCV
- AVX intrinsics for SIMD
- OpenMP for parallelization
