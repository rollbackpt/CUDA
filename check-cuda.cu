/*
Based on the hello-world created by Ingemar Ragnemalm 2010
(http://computer-graphics.se/hello-world-for-cuda.html)
and the book "CUDA by Example"

This example code detects CUDA devices, print their information
and tests the parallel programing using CUDA

Author: João Ribeiro

nvcc check-cuda.cu -L /usr/local/cuda/lib -lcudart -o check-cuda
*/

#include <stdio.h>
#include <unistd.h>

const int N = 16;
const int blocksize = 16;

__global__
void hello(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}

int main()
{
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	char *ad;
	int *bd;
	int dev_count;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);
	cudaDeviceProp prop;

	cudaGetDeviceCount(&dev_count);
	printf("Number of CUDA devices found: %d\n\n", dev_count);

	/* Get and print GPU information */
	for (int i = 0; i < dev_count; i++) {
		cudaGetDeviceProperties(&prop, i);

		printf( "--- General Information for device %d ---\n", i );
		printf( "Name: %s\n", prop.name );
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate: %d\n", prop.clockRate );
		printf( "Device copy overlap:" );

		if (prop.deviceOverlap)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );

		printf( "Kernel execition timeout :" );

		if (prop.kernelExecTimeoutEnabled)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );

		printf( "--- Memory Information for device %d ---\n", i );
		printf( "Total global mem: %ld\n", prop.totalGlobalMem );
		printf( "Total constant Mem: %ld\n", prop.totalConstMem );
		printf( "Max mem pitch: %ld\n", prop.memPitch );
		printf( "Texture Alignment: %ld\n", prop.textureAlignment );
		printf( "--- MP Information for device %d ---\n", i );
		printf( "Multiprocessor count: %d\n",prop.multiProcessorCount );
		printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
		printf( "Registers per mp: %d\n", prop.regsPerBlock );
		printf( "Threads in warp: %d\n", prop.warpSize );
		printf( "Max threads per block: %d\n", prop.maxThreadsPerBlock );
		printf( "Max thread dimensions: (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
		printf( "Max grid dimensions:(%d, %d, %d)\n",
		prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
		printf( "\n" );
	}
	/* End of print GPU information */

	printf("The next print will be the result of a parallel processed array. If you see the string \"Hello World!\" then CUDA is working!\n\n");
	printf("%s", a);

	/* Using CUDA to generate the string "World!"*/
	cudaMalloc( (void**)&ad, csize );
	cudaMalloc( (void**)&bd, isize );
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
	cudaFree( ad );
	cudaFree( bd );
	/* End of using CUDA to generate the string "World!"*/

	printf("%s\n\n", a);
	usleep(1000);
	return EXIT_SUCCESS;
}
