#include "stdio.h"
#include "cuda_runtime.h"
#include <math.h>
#include <stdlib.h>
#include <vector>
#include "helper_timer.h"
#include "helper_cuda.h"
#include "helper_string.h"
#define N 50000000
//#define NPARTITION 10 // tuned such that kernel takes a few microseconds

#define NCASCADING 10

#include <cstdlib>
void random_initialize(float* arr, size_t len)
{
	for (size_t idx = 0; idx < len; idx++)
	{
		arr[idx] = (std::rand() + 0.0) / RAND_MAX;
	}
	return;
}

inline void __checkCudaErrors(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		//exit(-1);

	}
	//return err;
}
#define checkCudaErrors(err) (__checkCudaErrors((err),__FILE__,__LINE__))

template <int NPARTITION>
__global__ void shortKernel(float* vector_d, float* in_d) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int curr_idx = idx; curr_idx < N / NPARTITION; curr_idx += blockDim.x * gridDim.x) {
		vector_d[curr_idx] = 1.23 * in_d[curr_idx];
	}
}
template <int NPARTITION>
int __main_01() {
	cudaGraph_t graph;
	cudaGraphExec_t instance;
	cudaStream_t stream;
	cudaKernelNodeParams kernelNodeParams;
	cudaGraphNode_t kernel_node[NCASCADING];
	float* input;
	input = (float*) malloc(sizeof(float) * N);
	random_initialize(input, N);
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	checkCudaErrors(cudaGraphCreate(&graph, 0));
	float* vector_d[NCASCADING + 1];
	for (int idx = 0; idx < NCASCADING + 1; idx++) {
		checkCudaErrors(cudaMalloc(&vector_d[idx], sizeof(float) * N));
	}
	checkCudaErrors(cudaMemcpy(vector_d[0], input,sizeof(float)*N, cudaMemcpyHostToDevice));
	StopWatchInterface* timerExec = NULL;
	sdkCreateTimer(&timerExec);
	sdkStartTimer(&timerExec);
//first iteration of ipartition: create graph then execute
	for (int iCascade = 0; iCascade < NCASCADING; iCascade++) {
		std::vector<cudaGraphNode_t> node_dependencies;
		if (iCascade != 0) {
			node_dependencies.push_back(kernel_node[iCascade - 1]);
		}
		void* kernelArgsPtr[2] = { (void*)&vector_d[iCascade+1],(void*)&vector_d[iCascade] };
		kernelNodeParams.func = (void*)shortKernel<NPARTITION>;
		kernelNodeParams.gridDim = 36;
		kernelNodeParams.blockDim = 1024;
		kernelNodeParams.kernelParams = (void**)&kernelArgsPtr;
		kernelNodeParams.extra = NULL;
		kernelNodeParams.sharedMemBytes = 0;
		cudaGraphAddKernelNode(&kernel_node[iCascade], graph, node_dependencies.data(), node_dependencies.size(), &kernelNodeParams);
		
	}
	checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
	checkCudaErrors(cudaGraphLaunch(instance, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));

	for (int ipartition = 1; ipartition < NPARTITION; ipartition++) {
		for (int iCascade = 0; iCascade < NCASCADING; iCascade++) {
			//replace parameter
			cudaKernelNodeParams kernelNodeParams_curr;
			float* kernelArgs_curr[2] = { &vector_d[iCascade + 1][N / NPARTITION * ipartition],&vector_d[iCascade][N / NPARTITION * ipartition] };
			void* kernelArgsPtr_curr[2] = { (void*)&kernelArgs_curr[0], (void*)&kernelArgs_curr[1] };
			kernelNodeParams_curr.func = (void*)shortKernel<NPARTITION>;
			kernelNodeParams_curr.gridDim = 36;
			kernelNodeParams_curr.blockDim = 1024;
			kernelNodeParams_curr.kernelParams = (void**)&kernelArgsPtr_curr;
			kernelNodeParams_curr.extra = NULL;
			kernelNodeParams_curr.sharedMemBytes = 0;
#if __CUDA_ARCH__ >= 800

#endif
			checkCudaErrors(cudaGraphExecKernelNodeSetParams(instance, kernel_node[iCascade], &kernelNodeParams_curr));
		}
		checkCudaErrors(cudaGraphLaunch(instance, stream));
		checkCudaErrors(cudaStreamSynchronize(stream));
	}

	sdkStopTimer(&timerExec);
	printf("Execution time: %f (ms)\n", sdkGetTimerValue(&timerExec));
	checkCudaErrors(cudaGraphExecDestroy(instance));
	checkCudaErrors(cudaGraphDestroy(graph));
	checkCudaErrors(cudaStreamDestroy(stream));
	return 0;
}

int main0() {
	return __main_01<250>();
}

int main1() {
	return __main_01<1>();
}

int main2() {
	return 0;
}

int main(int argc, char** argv) {
	if (checkCmdLineFlag(argc, (const char**)argv, "help")) {
		printf("Command line: jacobiCudaGraphs [-option]\n");
		printf("Valid options:\n");
		printf(
			"-gpumethod=<0,1 or 2>  : 0 - [Default] CUDA Graph Paritioned");
		printf("                       : 1 - CUDA Graph\n");
		printf("                       : 2 - Non CUDA Graph\n");
	}
	int gpumethod = 0;
	if (checkCmdLineFlag(argc, (const char**)argv, "gpumethod")) {
		gpumethod = getCmdLineArgumentInt(argc, (const char**)argv, "gpumethod");
		if (gpumethod < 0 || gpumethod > 2) {
			printf("Error: gpumethod must be 0 or 1 or 2, gpumethod = %d is invalid\n", gpumethod);
			exit(EXIT_SUCCESS);
		}

	}
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	switch (gpumethod) {
	case 0:
		main0();
		break;
	case 1:
		main1();
		break;
	case 2:
		main2();
		break;
	}
	sdkStopTimer(&timer);
	printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
}