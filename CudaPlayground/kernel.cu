#include "stdio.h"
#include "cuda_runtime.h"
#include <math.h>
#include <stdlib.h>
#include <vector>
#include "helper_timer.h"
#include "helper_cuda.h"
#include "helper_string.h"
#if MY_CUDA_ARCH_IDENTIFIER >= 800 // assuming 3090
#define N 687865856
#define NUM_CASCADING 8
#define NUM_PARTITION 512 // each arry occupies 1.28125MB
#define GRIDDIM 82
#define BLOCK_THREADNUM 1024
#else
#define N 50000000 //assuming 2070max-q
#define NUM_CASCADING 10
#define NUM_PARTITION 150 // tuned such that kernel takes a few microseconds
#define GRIDDIM 36
#endif

#include <cstdlib>
#include <cmath>

template <int NCASCADING, int NLEN>
int verify_saxpy(float *output, float *input)
{
	float timer = pow(1.23, NCASCADING);
	int result = 0;
	for (size_t idx = 0; idx < NLEN; idx++)
	{
		if (abs(output[idx] - input[idx] * timer) > 0.000002)
		{
			printf("Error: %lu %f %f!\n", idx, output[idx], input[idx] * timer);
			result++;
		}
	}
	return result;
}

void random_initialize(float *arr, size_t len)
{
	for (size_t idx = 0; idx < len; idx++)
	{
		arr[idx] = (std::rand() + 0.0) / RAND_MAX;
	}
	return;
}

inline void __checkCudaErrors(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		//exit(-1);
	}
	//return err;
}
#define checkCudaErrors(err) (__checkCudaErrors((err), __FILE__, __LINE__))

template <int NPARTITION, int NLEN>
__device__ __forceinline__ void shortKernel_nonprefetch(float *__restrict__ vector_d, const float *__restrict__ in_d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//#pragma unroll
	for (int curr_idx = idx; curr_idx < NLEN / NPARTITION; curr_idx += blockDim.x * gridDim.x)
	{
		//__stcg(&vector_d[curr_idx], 1.23 * __ldlu(&in_d[curr_idx]));
		vector_d[curr_idx] = 1.23 * in_d[curr_idx];
	}
}
#include <cooperative_groups.h>
#include <cuda/barrier>
template <int NPARTITION, int NLEN>
__device__ __forceinline__ void shortKernel_prefetch_2(float *__restrict__ vector_d, const float *__restrict__ in_d)
{
	auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float shared_storage[BLOCK_THREADNUM * 2];
	__shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier[2];
	if (block.thread_rank() == 0)
	{
		init(&barrier[0], block.size());
		init(&barrier[1], block.size());
	}
	block.sync();
	cuda::memcpy_async(block, shared_storage, in_d, sizeof(float) * block.size(), barrier[0]);
	//shared_storage[idx]=in_d[idx];
	//float temp[2];
	//temp[0]=in_d[idx];
	int ping_pong = 0;
	int curr_idx = idx;
	for (int iter_idx = 1; curr_idx < NLEN / NPARTITION - BLOCK_THREADNUM * GRIDDIM; curr_idx += BLOCK_THREADNUM * GRIDDIM, iter_idx++)
	{
		//vector_d[curr_idx]=1.23*temp[ping_pong];
		ping_pong++;
		cuda::memcpy_async(block, shared_storage + BLOCK_THREADNUM * (ping_pong % 2), in_d + (BLOCK_THREADNUM * GRIDDIM * iter_idx), sizeof(float) * block.size(), barrier[ping_pong % 2]);
		barrier[(ping_pong - 1) % 2].arrive_and_wait();
		vector_d[curr_idx] = 1.23 * shared_storage[threadIdx.x + ((ping_pong - 1) % 2) * BLOCK_THREADNUM];
		//shared_storage[idx%BLOCK_THREADNUM+(1-ping_pong%2)*BLOCK_THREADNUM]=in_d[curr_idx+BLOCK_THREADNUM];
		//temp[(++ping_pong)%2]=in_d[curr_idx+BLOCK_THREADNUM];
	}
	barrier[(ping_pong) % 2].arrive_and_wait();
	//vector_d[curr_idx+BLOCK_THREADNUM]=1.23*temp[ping_pong];
	vector_d[curr_idx] = 1.23 * shared_storage[threadIdx.x + (ping_pong % 2) * BLOCK_THREADNUM];
}

template <int NPARTITION, int NLEN>
__device__ __forceinline__ void shortKernel_prefetch(float *__restrict__ vector_d, const float *__restrict__ in_d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
#define NUM_PINGPONG 16
	__shared__ float shared_storage[BLOCK_THREADNUM * 2];
	//shared_storage[threadIdx.x]=in_d[idx];
	float temp[NUM_PINGPONG];
	temp[0] = in_d[idx];
	int ping_pong = 0;
	int curr_idx = idx;
#pragma unroll 16
	for (; curr_idx < NLEN / NPARTITION - BLOCK_THREADNUM * GRIDDIM; curr_idx += BLOCK_THREADNUM * GRIDDIM)
	{
		//vector_d[curr_idx]=1.23*temp[ping_pong];
		ping_pong = (1 + ping_pong) % NUM_PINGPONG;
		//shared_storage[threadIdx.x+ping_pong*BLOCK_THREADNUM]=in_d[curr_idx+blockIdx.x * blockDim.x];
		temp[ping_pong] = in_d[curr_idx + BLOCK_THREADNUM * GRIDDIM];
		vector_d[curr_idx] = 1.23 * temp[(ping_pong + NUM_PINGPONG - 1) % NUM_PINGPONG]; //1.23*shared_storage[threadIdx.x+(1-ping_pong)*BLOCK_THREADNUM];
	}
	vector_d[curr_idx] = 1.23 * temp[ping_pong];
	//vector_d[curr_idx]=1.23*shared_storage[threadIdx.x+ping_pong*BLOCK_THREADNUM];
}

template <int NPARTITION, int NLEN, bool FLAG_PREFETCH>
__global__ void shortKernel(float *__restrict__ vector_d, const float *__restrict__ in_d)
{
	//__builtin_assume(blockDim.x*gridDim.x==BLOCK_THREADNUM);
	if constexpr (FLAG_PREFETCH)
	{
		shortKernel_prefetch<NPARTITION, NLEN>(vector_d, in_d);
	}
	else
	{
		shortKernel_nonprefetch<NPARTITION, NLEN>(vector_d, in_d);
	}
}

template <int NPARTITION, int NCASCADING, int NLEN>
__global__ void shortKernel_merged(float *vectors_d[NCASCADING + 1], const int ipartition)
{
	//__builtin_assume(blockDim.x*gridDim.x==BLOCK_THREADNUM);

	int idx = blockIdx.x * blockDim.x + threadIdx.x + NLEN / NPARTITION * ipartition;
#pragma unroll
	for (int i_cascading = 0; i_cascading < NCASCADING; i_cascading++)
	{
		for (int curr_idx = idx; curr_idx < NLEN / NPARTITION * (ipartition + 1); curr_idx += blockDim.x * gridDim.x)
		{
			vectors_d[i_cascading + 1][curr_idx] = 1.23 * vectors_d[i_cascading][curr_idx];
		}
	}
}

template <int NPARTITION, int NCASCADING, int NLEN>
__global__ void shortKernel_merged_optimized(float *vectors_d[NCASCADING + 1], const int ipartition)
{
	//__builtin_assume(blockDim.x*gridDim.x==BLOCK_THREADNUM);
	long long idx = blockIdx.x * blockDim.x + threadIdx.x + NLEN / NPARTITION * ipartition;
#pragma unroll
	for (int i_cascading = 0; i_cascading < NCASCADING; i_cascading++)
	{
#pragma unroll
		for (int curr_idx = idx; curr_idx < NLEN / NPARTITION * (ipartition + 1); curr_idx += blockDim.x * gridDim.x)
		{
			__stcg(&vectors_d[i_cascading + 1][curr_idx], 1.23 * __ldlu(&vectors_d[i_cascading][curr_idx]));
		}
	}
}

struct param_resetStreamAccessPolicyWindow
{
	struct cudaAccessPolicyWindow accessPolicyWindow;
	cudaStream_t stream;
};

void resetStreamAccessPolicyWindow(void *param)
{
	struct cudaAccessPolicyWindow accessPolicyWindow = ((struct param_resetStreamAccessPolicyWindow *)param)->accessPolicyWindow;
	cudaStream_t stream = ((struct param_resetStreamAccessPolicyWindow *)param)->stream;

	cudaStreamAttrValue attr;
	attr.accessPolicyWindow.base_ptr = accessPolicyWindow.base_ptr;
	attr.accessPolicyWindow.num_bytes = accessPolicyWindow.num_bytes;
	// hitRatio causes the hardware to select the memory window to designate as persistent in the area set-aside in L2
	attr.accessPolicyWindow.hitRatio = accessPolicyWindow.hitRatio;
	// Type of access property on cache hit
	attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
	// Type of access property on cache miss
	attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
	checkCudaErrors(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

template <int NPARTITION, int NCASCADING, bool FLAG_ENABLE_L2_POLICY>
int __main_01()
{
	cudaGraph_t graph;
	cudaGraphExec_t instance;
	cudaStream_t stream;
	cudaKernelNodeParams kernelNodeParams;
	cudaGraphNode_t kernel_node[NCASCADING];
	cudaGraphNode_t host_nodes[NCASCADING - 1];
	float *input;
	input = (float *)malloc(sizeof(float) * N);
	random_initialize(input, N);
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	checkCudaErrors(cudaGraphCreate(&graph, 0));
	float *vector_d[NCASCADING + 1];
	for (int idx = 0; idx < NCASCADING + 1; idx++)
	{
		checkCudaErrors(cudaMalloc(&vector_d[idx], sizeof(float) * N));
	}
	checkCudaErrors(cudaMemcpy(vector_d[0], input, sizeof(float) * N, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaStreamSynchronize(stream));
	StopWatchInterface *timerExec = NULL;
	sdkCreateTimer(&timerExec);
	sdkStartTimer(&timerExec);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//first iteration of ipartition: create graph then execute
	for (int iCascade = 0; iCascade < NCASCADING; iCascade++)
	{
		std::vector<cudaGraphNode_t> node_dependencies;
		if (iCascade != 0)
		{
#if MY_CUDA_ARCH_IDENTIFIER >= 800
#ifdef HOST_NODE_CANNOT_EXECUTE_CUDA_FUNCTION
			if constexpr (FLAG_ENABLE_L2_POLICY)
			{
				cudaHostNodeParams hostNodeParams;
				hostNodeParams.fn = resetStreamAccessPolicyWindow;
				struct param_resetStreamAccessPolicyWindow host_params;
				cudaKernelNodeAttrValue last_kernel_node_attribute;
				checkCudaErrors(cudaGraphKernelNodeGetAttribute(kernel_node[iCascade - 1], cudaKernelNodeAttributeAccessPolicyWindow, &last_kernel_node_attribute));
				host_params.accessPolicyWindow = last_kernel_node_attribute.accessPolicyWindow;
				host_params.stream = stream;
				hostNodeParams.userData = (void *)&host_params;
				std::vector<cudaGraphNode_t> host_node_dependencies = {kernel_node[iCascade - 1]};
				checkCudaErrors(cudaGraphAddHostNode(&host_nodes[iCascade - 1], graph, host_node_dependencies.data(), host_node_dependencies.size(), &hostNodeParams));
				node_dependencies.push_back(host_nodes[iCascade - 1]);
			}
			else
			{
#endif
				node_dependencies.push_back(kernel_node[iCascade - 1]);
#ifdef HOST_NODE_CANNOT_EXECUTE_CUDA_FUNCTION
			}
#endif
#else
			node_dependencies.push_back(kernel_node[iCascade - 1]);
#endif
		}
		void *kernelArgsPtr[2] = {(void *)&vector_d[iCascade + 1], (void *)&vector_d[iCascade]};
		kernelNodeParams.func = (void *)shortKernel<NPARTITION, N, false>;
		kernelNodeParams.gridDim = GRIDDIM;
		kernelNodeParams.blockDim = BLOCK_THREADNUM;
		kernelNodeParams.kernelParams = (void **)&kernelArgsPtr;
		kernelNodeParams.extra = NULL;
		kernelNodeParams.sharedMemBytes = 0;
		checkCudaErrors(cudaGraphAddKernelNode(&kernel_node[iCascade], graph, node_dependencies.data(), node_dependencies.size(), &kernelNodeParams));
	}
	checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
	checkCudaErrors(cudaGraphLaunch(instance, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	for (int ipartition = 1; ipartition < NPARTITION; ipartition++)
	{
		for (int iCascade = 0; iCascade < NCASCADING; iCascade++)
		{
			//replace parameter
			cudaKernelNodeParams kernelNodeParams_curr;
			float *kernelArgs_curr[2] = {&vector_d[iCascade + 1][N / NPARTITION * ipartition], &vector_d[iCascade][N / NPARTITION * ipartition]};
			void *kernelArgsPtr_curr[2] = {(void *)&kernelArgs_curr[0], (void *)&kernelArgs_curr[1]};
			kernelNodeParams_curr.func = (void *)shortKernel<NPARTITION, N, false>;
			kernelNodeParams_curr.gridDim = GRIDDIM;
			kernelNodeParams_curr.blockDim = BLOCK_THREADNUM;
			kernelNodeParams_curr.kernelParams = (void **)&kernelArgsPtr_curr;
			kernelNodeParams_curr.extra = NULL;
			kernelNodeParams_curr.sharedMemBytes = 0;
#if MY_CUDA_ARCH_IDENTIFIER >= 800
			if constexpr (FLAG_ENABLE_L2_POLICY)
			{
				cudaKernelNodeAttrValue node_attribute;																						 // Kernel level attributes data structure
				node_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void *>(&vector_d[iCascade + 1][N / NPARTITION * ipartition]); // Global Memory data pointer
				node_attribute.accessPolicyWindow.num_bytes = N / NPARTITION * sizeof(float);												 // Number of bytes for persistence access.
																																			 // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
				node_attribute.accessPolicyWindow.hitRatio = 0.6;																			 // Hint for cache hit ratio
				node_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;													 // Type of access property on cache hit
				node_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;													 // Type of access property on cache miss.

				//Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
				checkCudaErrors(cudaGraphKernelNodeSetAttribute(kernel_node[iCascade], cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute));
				checkCudaErrors(cudaGraphExecKernelNodeSetParams(instance, kernel_node[iCascade], &kernelNodeParams_curr));
#ifdef HOST_NODE_CANNOT_EXECUTE_CUDA_FUNCTION
				//TODO: set graph host node attribute
				if (iCascade != NCASCADING)
				{
					struct param_resetStreamAccessPolicyWindow params_host_curr;
					params_host_curr.stream = stream;
					params_host_curr.accessPolicyWindow = node_attribute.accessPolicyWindow;
					cudaHostNodeParams hostNodeParams;
					hostNodeParams.fn = resetStreamAccessPolicyWindow;
					hostNodeParams.userData = (void *)&params_host_curr;
					cudaGraphHostNodeSetParams(host_nodes[iCascade], &hostNodeParams);
				}
#endif
			}
			else
			{
				checkCudaErrors(cudaGraphExecKernelNodeSetParams(instance, kernel_node[iCascade], &kernelNodeParams_curr));
			}
#else
			checkCudaErrors(cudaGraphExecKernelNodeSetParams(instance, kernel_node[iCascade], &kernelNodeParams_curr));
#endif
		}
		checkCudaErrors(cudaGraphLaunch(instance, stream));
		checkCudaErrors(cudaStreamSynchronize(stream));
	}
	cudaEventRecord(stop);

	sdkStopTimer(&timerExec);
	printf("Execution time: %f (ms)\n", sdkGetTimerValue(&timerExec));
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Execution time (GPU): %f (ms)\n", milliseconds);
	checkCudaErrors(cudaGraphExecDestroy(instance));
	checkCudaErrors(cudaGraphDestroy(graph));
	checkCudaErrors(cudaStreamDestroy(stream));
	float *output = (float *)malloc(sizeof(float) * N);
	checkCudaErrors(cudaMemcpy(output, vector_d[NCASCADING], sizeof(float) * N, cudaMemcpyDeviceToHost));
	printf("Errors: %d\n", verify_saxpy<NCASCADING, N>(output, input));
	return 0;
}

int main0()
{
	cudaFuncSetCacheConfig(shortKernel<NUM_PARTITION, N, false>, cudaFuncCachePreferShared);
	return __main_01<NUM_PARTITION, NUM_CASCADING, false>();
}

int main1()
{
	cudaFuncSetCacheConfig(shortKernel<1, N, false>, cudaFuncCachePreferShared);
	return __main_01<1, NUM_CASCADING, false>();
}

int main3()
{
	cudaFuncSetCacheConfig(shortKernel<NUM_PARTITION, N, false>, cudaFuncCachePreferShared);
	return __main_01<NUM_PARTITION, NUM_CASCADING, true>();
}

int main4()
{
	cudaFuncSetCacheConfig(shortKernel<1, N, false>, cudaFuncCachePreferShared);
	return __main_01<1, NUM_CASCADING, true>();
}

template <int NPARTITION, int NCASCADING, bool FLAG_OPTIMIZATION, bool FLAG_BASELINE>
int __main2()
{
	float *input;
	input = (float *)malloc(sizeof(float) * N);
	random_initialize(input, N);
	float *vectors_d[NCASCADING + 1];
	float **vectors_d_d;
	for (int idx = 0; idx < NCASCADING + 1; idx++)
	{
		checkCudaErrors(cudaMalloc(&vectors_d[idx], sizeof(float) * N));
	}
	checkCudaErrors(cudaMalloc(&vectors_d_d, sizeof(float *) * (NCASCADING + 1)));
	checkCudaErrors(cudaMemcpy(vectors_d[0], input, sizeof(float) * N, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(vectors_d_d, vectors_d, sizeof(float *) * (NCASCADING + 1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaStreamSynchronize(0));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	StopWatchInterface *timerExec = NULL;
	sdkCreateTimer(&timerExec);
	sdkStartTimer(&timerExec);
	cudaEventRecord(start);
	for (int ipartition = 0; ipartition < NPARTITION; ipartition++)
	{
		if constexpr (FLAG_BASELINE)
		{
			for (int i_cascading = 0; i_cascading < NCASCADING; i_cascading++)
			{
				shortKernel<NPARTITION, N, true><<<GRIDDIM, BLOCK_THREADNUM>>>(&vectors_d[i_cascading + 1][N / NPARTITION * ipartition], &vectors_d[i_cascading + 0][N / NPARTITION * ipartition]);
			}
		}
		else
		{
			if constexpr (FLAG_OPTIMIZATION)
			{
				shortKernel_merged_optimized<NPARTITION, NUM_CASCADING, N><<<GRIDDIM, BLOCK_THREADNUM>>>(vectors_d_d, ipartition);
			}
			else
			{
				shortKernel_merged<NPARTITION, NUM_CASCADING, N><<<GRIDDIM, BLOCK_THREADNUM>>>(vectors_d_d, ipartition);
			}
		}
	}
	cudaEventRecord(stop);
	checkCudaErrors(cudaStreamSynchronize(0));
	sdkStopTimer(&timerExec);
	printf("Execution time: %f (ms)\n", sdkGetTimerValue(&timerExec));
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Execution time (GPU): %f (ms)\n", milliseconds);
	float *output = (float *)malloc(sizeof(float) * N);
	checkCudaErrors(cudaMemcpy(output, vectors_d[NCASCADING], sizeof(float) * N, cudaMemcpyDeviceToHost));
	printf("Errors: %d\n", verify_saxpy<NCASCADING, N>(output, input));
	return 0;
}

int main2()
{
	cudaFuncSetCacheConfig(shortKernel_merged<NUM_PARTITION, NUM_CASCADING, N>, cudaFuncCachePreferShared);
	return __main2<NUM_PARTITION, NUM_CASCADING, false, false>();
}

int main5()
{
	cudaFuncSetCacheConfig(shortKernel_merged<NUM_PARTITION, NUM_CASCADING, N>, cudaFuncCachePreferShared);
	return __main2<NUM_PARTITION, NUM_CASCADING, true, false>();
}

int main6()
{
	//cudaFuncSetCacheConfig(shortKernel<NUM_PARTITION, NUM_CASCADING, N, false>, cudaFuncCachePreferShared);
	return __main2<NUM_PARTITION, NUM_CASCADING, false, true>();
}

int main7()
{
	//cudaFuncSetCacheConfig(shortKernel<1, NUM_CASCADING, N, false>, cudaFuncCachePreferShared);
	return __main2<1, NUM_CASCADING, false, false>();
}

int main(int argc, char **argv)
{
#if MY_CUDA_ARCH_IDENTIFIER >= 800
	printf("cuda arch >= 800\n");
#endif
	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		printf("Command line: jacobiCudaGraphs [-option]\n");
		printf("Valid options:\n");
		printf(
			"-gpumethod=<0,1 or 2>  : 0 - [Default] CUDA Graph Paritioned");
		printf("                       : 1 - CUDA Graph\n");
		printf("                       : 2 - Non CUDA Graph\n");
	}
	int gpumethod = -1;
	if (checkCmdLineFlag(argc, (const char **)argv, "gpumethod"))
	{
		gpumethod = getCmdLineArgumentInt(argc, (const char **)argv, "gpumethod");
		if (gpumethod < 0 || gpumethod > 7)
		{
			printf("Error: gpumethod must be 0 or 1 or 2 or 3 or 4, gpumethod = %d is invalid\n", gpumethod);
			exit(EXIT_SUCCESS);
		}
	}
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	switch (gpumethod)
	{
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
	case 5:
		main5();
		break;
	case 6:
		main6();
		break;
	case 7:
		main7();
		break;
	}
	sdkStopTimer(&timer);
	printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
}