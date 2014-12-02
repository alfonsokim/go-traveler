#include <stdio.h>

inline void GPUassert(cudaError_t code, char * file, int line, bool Abort=true)
{
    if (code != 0) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
        if (Abort) exit(code);
    }       
}

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }
#define NUMPATHS 10

__host__ __device__ void printArray(int *a, int lenght){
    for(int i = 0; i < lenght; i++){
        printf("%i, ", a[i]);
    }
}

__host__ __device__ void swap(int *x, int *y){
    int temp;
    temp = *x;
    *x = *y;
    *y = temp;
}

__device__ void permute_device(int *a, int i, int n, int tid, int* count) {
    if (i == n) { 
        //int* perm = a - 1; 
        //printf("Permutation nr. %i from thread nr. %i", count[0], tid);
        printArray(a, 10);
        printf("\n");
        int* result = a - 1; 
        int distance = 0;
        for(int i = 0; i < NUMPATHS-1; i++){
            distance += result[i];
        }
        printf("Permutation nr. %i from thread nr. %i distance = %i\n", count[0], tid, distance);
        count[0] = count[0] + 1; 
    } else {
        for (int j = i; j <= n; j++) {
            swap((a+i), (a+j));
            permute_device(a, i+1, n, tid, count);
            swap((a+i), (a+j)); //backtrack
        }
    }
} 

__global__ void permute_kernel(int* d_A, int size) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int count[1]; 
    count[0] = 0;

    int local_array[10];

    for (int i=0; i<size; i++){
        local_array[i] = d_A[i];  
    } 

    swap(local_array + threadIdx.x, local_array);
    permute_device(local_array+1, 0, 5, tid, count);

}

int factorial(int i) {
    int result = 1;
    while(i > 0){
        result *= i;
        i--;
    }
    return result;
}

int main(){

    int h_a[10] = { 20, 5, 14, 9, 16, 19, 11, 7, 13, 2 };

    int* d_a; 
    cudaMalloc((void**)&d_a, sizeof(h_a));

    GPUerrchk(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice));

    printf("\n\n Permutations on GPU\n");
    permute_kernel<<<1,10>>>(d_a, 10);
    GPUerrchk(cudaPeekAtLastError());
    GPUerrchk(cudaDeviceSynchronize());

    getchar();
    return 0;
}

