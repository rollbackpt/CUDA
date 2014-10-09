This code is based on the hello-world created by Ingemar Ragnemalm 2010 (http://computer-graphics.se/hello-world-for-cuda.html) and the book "CUDA by Example".

This example code detects CUDA devices, print their information and tests the parallel programing using CUDA.

Compile:
```
nvcc check-cuda.cu -L /usr/local/cuda/lib -lcudart -o check-cuda
```

Run:
```
./check-cuda
```
