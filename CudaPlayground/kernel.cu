﻿#include "stdio.h"
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
		__stcg(&vector_d[curr_idx], 1.23 * __ldlu(&in_d[curr_idx]));
	}
}

template <int NPARTITION>
__global__ void shortKernel_merged(float** vectors_d, int n_cascading) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i_cascading = 0; i_cascading < n_cascading; i_cascading++) {
		for (int curr_idx = idx; curr_idx < N / NPARTITION; curr_idx += blockDim.x * gridDim.x) {
			vectors_d[i_cascading+1][curr_idx] = 1.23 * vectors_d[i_cascading][curr_idx];
		}
	}
}

struct param_resetStreamAccessPolicyWindow {
	struct cudaAccessPolicyWindow accessPolicyWindow;
	cudaStream_t stream;
};

void resetStreamAccessPolicyWindow(void* param) {
	struct cudaAccessPolicyWindow accessPolicyWindow = ((struct param_resetStreamAccessPolicyWindow*)param)->accessPolicyWindow;
	cudaStream_t stream = ((struct param_resetStreamAccessPolicyWindow*)param)->stream;

	cudaStreamAttrValue attr;
	attr.accessPolicyWindow.base_ptr = accessPolicyWindow.base_ptr;
	attr.accessPolicyWindow.num_bytes = accessPolicyWindow.num_bytes;
	// hitRatio causes the hardware to select the memory window to designate as persistent in the area set-aside in L2
	attr.accessPolicyWindow.hitRatio = accessPolicyWindow.hitRatio;
	// Type of access property on cache hit
	attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
	// Type of access property on cache miss
	attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
	cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}

template <int NPARTITION, bool FLAG_ENABLE_L2_POLICY>
int __main_01() {
	cudaGraph_t graph;
	cudaGraphExec_t instance;
	cudaStream_t stream;
	cudaKernelNodeParams kernelNodeParams;
	cudaGraphNode_t kernel_node[NCASCADING];
	cudaGraphNode_t host_nodes[NCASCADING - 1];
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
	checkCudaErrors(cudaStreamSynchronize(stream));
	StopWatchInterface* timerExec = NULL;
	sdkCreateTimer(&timerExec);
	sdkStartTimer(&timerExec);
//first iteration of ipartition: create graph then execute
	for (int iCascade = 0; iCascade < NCASCADING; iCascade++) {
		std::vector<cudaGraphNode_t> node_dependencies;
		if (iCascade != 0) {
#if __CUDA_ARCH__ >= 800
			if constexpr (FLAG_ENABLE_L2_POLICY) {
				cudaHostNodeParams hostNodeParams;
				hostNodeParams.fn = resetStreamAccessPolicyWindow;
				struct param_resetStreamAccessPolicyWindow host_params;
				cudaKernelNodeAttrValue last_kernel_node_attribute;
				cudaGraphKernelNodeGetAttribute(kernel_node[iCascade - 1], cudaKernelNodeAttributeAccessPolicyWindow, &last_kernel_node_attribute);
				host_params.accessPolicyWindow = last_kernel_node_attribute.accessPolicyWindow;
				host_params.stream = stream;
				hostNodeParams.userData = (void*)&host_params;
				std::vector<cudaGraphNode_t> host_node_dependencies = { kernel_node[iCascade - 1] };
				cudaGraphAddHostNode(&host_nodes[iCascade - 1], graph, host_node_dependencies.data(), host_node_dependencies.size(), &hostNodeParams);
				node_dependencies.push_back(host_nodes[iCascade - 1]);
			}
			else {
				node_dependencies.push_back(kernel_node[iCascade - 1]);
			}
#else
			node_dependencies.push_back(kernel_node[iCascade - 1]);
#endif
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
			if constexpr(FLAG_ENABLE_L2_POLICY){
				cudaKernelNodeAttrValue node_attribute;                                     // Kernel level attributes data structure
				node_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(&vector_d[iCascade + 1][N / NPARTITION * ipartition]); // Global Memory data pointer
				node_attribute.accessPolicyWindow.num_bytes = N / NPARTITION & sizeof(float);                    // Number of bytes for persistence access.
																							// (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
				node_attribute.accessPolicyWindow.hitRatio = 0.6;                          // Hint for cache hit ratio
				node_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Type of access property on cache hit
				node_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

				//Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
				cudaGraphKernelNodeSetAttribute(kernel_node[iCascade], cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);

				//TODO: set graph host node attribute
				if (iCascade != NCASCADING) {
					struct param_resetStreamAccessPolicyWindow params_host_curr;
					params_host_curr.stream = stream;
					params_host_curr.accessPolicyWindow = node_attribute.accessPolicyWindow;
					cudaHostNodeParams hostNodeParams;
					hostNodeParams.fn = resetStreamAccessPolicyWindow;
					hostNodeParams.userData = (void*)&params_host_curr;
					cudaGraphHostNodeSetParams(host_nodes[iCascade], &hostNodeParams);
				}
			}
			else {
				checkCudaErrors(cudaGraphExecKernelNodeSetParams(instance, kernel_node[iCascade], &kernelNodeParams_curr));
			}
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
	cudaFuncSetCacheConfig(shortKernel<125>, cudaFuncCachePreferShared);
	return __main_01<125, false>();
}

int main1() {
	cudaFuncSetCacheConfig(shortKernel<1>, cudaFuncCachePreferShared);
	return __main_01<1, false>();
}

int main3() {
	cudaFuncSetCacheConfig(shortKernel<125>, cudaFuncCachePreferShared);
	return __main_01<125, true>();
}

int main4() {
	cudaFuncSetCacheConfig(shortKernel<1>, cudaFuncCachePreferShared);
	return __main_01<1, true>();
}

template <int NPARTITION>
int __main2() {
	float* input;
	input = (float*)malloc(sizeof(float) * N);
	random_initialize(input, N);
	float* vectors_d[NCASCADING + 1];
	for (int idx = 0; idx < NCASCADING + 1; idx++) {
		checkCudaErrors(cudaMalloc(&vectors_d[idx], sizeof(float) * N));
	}
	checkCudaErrors(cudaMemcpy(vectors_d[0], input, sizeof(float) * N, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaStreamSynchronize(0));
	StopWatchInterface* timerExec = NULL;
	sdkCreateTimer(&timerExec);
	sdkStartTimer(&timerExec);
	for (int ipartition = 1; ipartition < NPARTITION; ipartition++) {

		shortKernel_merged<NPARTITION><<<36,1024>>>(vectors_d,NCASCADING);
	}
	checkCudaErrors(cudaStreamSynchronize(0));
	sdkStopTimer(&timerExec);
	printf("Execution time: %f (ms)\n", sdkGetTimerValue(&timerExec));
	return 0;
}

int main2() {
	cudaFuncSetCacheConfig(shortKernel_merged<500>, cudaFuncCachePreferShared);
	return __main2<125>();
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
		if (gpumethod < 0 || gpumethod > 4) {
			printf("Error: gpumethod must be 0 or 1 or 2 or 3 or 4, gpumethod = %d is invalid\n", gpumethod);
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
	case 3:
		main3();
		break;
	case 4:
		main4();
		break;
	}
	sdkStopTimer(&timer);
	printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
}