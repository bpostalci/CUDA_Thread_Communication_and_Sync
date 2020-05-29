// Berat Postalcioglu
/* OUTPUT

	minimum element of the array (minCPU): -9649.35
	minimum element of the array (minGPU): -9649.35

*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <cstdio>
#include <ctime>

const int ArrSize = 60000;
const int ThreadsPerBlock = 512;
const int BlocksPerGrid = 32;

// generates a random array
void generateArray(double *data, int count) {
	//generate a random data set
	for (int i = 0; i < count; i++) {
		data[i] = rand() / (rand() + 1.1) * (rand() % 2 ? 1 : -1);
	}

}

double minCPU(double *data, int count)
{
	int minIndex = 0;
	for (int i = 0; i < count; i++)
	{
		if (std::isgreater(data[minIndex], data[i]))
		{
			minIndex = i;
		}
	}
	return data[minIndex];
}

__global__ void minGPU(double *data, int count, double *res)
{
	__shared__ double cache[ThreadsPerBlock];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	double temp = 0;

	while (tid < count)
	{
		temp += data[tid];
		//cache[cacheIndex] = data[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = temp;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
		{
			if (cache[cacheIndex] > cache[cacheIndex + i])
			{
				cache[cacheIndex] = cache[cacheIndex + i];
			}
		}			
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0)
		res[blockIdx.x] = cache[0];
	
}

int main()
{
	srand(time(NULL));

	// cpu
	double data[ArrSize];
	generateArray(data, ArrSize);
	double minElementCpu = minCPU(data, ArrSize);
	printf("minimum element of the array (minCPU): %.2f\n", minElementCpu);

	// gpu
	double *gpuData, *gpuRes;
	cudaMalloc((void**)&gpuData, ArrSize * sizeof(double));
	cudaMalloc((void**)&gpuRes, BlocksPerGrid * sizeof(double));
	cudaMemcpy((void*)gpuData, (const void*) data, ArrSize * sizeof(double), cudaMemcpyHostToDevice);
	minGPU <<<BlocksPerGrid, ThreadsPerBlock>>> (gpuData, ArrSize, gpuRes);

	double blockResults[BlocksPerGrid];
	cudaMemcpy((void*)blockResults, (const void *)gpuRes, BlocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

	double minElementGpu = minCPU(blockResults, BlocksPerGrid);
	printf("minimum element of the array (minGPU): %.2f\n", minElementGpu);
	
	return 0;
}