#include <stdio.h>

struct City {
    int x, y;
};

inline void GPUassert(cudaError_t code, char * file, int line, bool Abort=true)
{
    if (code != 0) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
        if (Abort) exit(code);
    }       
}

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }

#define NUMCITIES 10
#define CITYSIZE sizeof(struct City)
// #define CITYSIZE 1


__host__ __device__ void printCity(struct City c){
    printf("[x=%i, y=%i]\n", c.x, c.y);
}

__host__ __device__ void swap(struct City *a, int x, int y){
    struct City temp;
    temp = a[x];
    a[x] = a[y];
    a[y] = temp;
}

__device__ double get_distance(struct City c1, struct City c2){
    double x = double(c1.x - c2.x);
    double y = double(c1.y - c2.y);
    return sqrt((x*x) + (y*y));
}

__device__ long total_distance(struct City *path){
    long distance = 0;
    for(int i = 0; i < NUMCITIES - 1; i++){
        distance += get_distance(path[i], path[i+1]);
    }
    return (long)distance;
}

__device__ void shortest_path(struct City *path){
    int best_path_idx = 0;
    for(int i = 0; i < NUMCITIES - 1; i++){
        printCity(path[i]);
    }
}

__device__ void permutations_kernel(struct City *a, int i, int length, int tid) { 

    if (length == i){
        long distance = total_distance(a);
    } else {
        for (int j = i; j < length; j++) {
            swap(a, i, j);
            // CUDA
            // permutations(a, i+1, length, tid, count);
            permutations_kernel(a, i+1, length, tid);
            swap(a, i, j);
        }
    }
}

/*
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
*/

__global__ void permute_kernel(struct City *dev_cities, int size) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int count[1]; 
    count[0] = 0;

    struct City local_array[NUMCITIES];

    for (int i=0; i<size; i++){
        local_array[i] = dev_cities[i];  
    } 

    //swap(local_array + threadIdx.x, local_array);
    //swap(local_array, threadIdx.x, 0);
    permutations_kernel(local_array, 0, NUMCITIES, tid);

}

int main(){

    struct City host_cities[NUMCITIES];
    for(int c = 0; c < NUMCITIES; c++){
        host_cities[c].x = rand() % 20 + 5;
        host_cities[c].y = rand() % 20 + 5;
    }

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    struct City *device_cities; 
    cudaMalloc((void**)&device_cities, sizeof(host_cities));
    GPUerrchk(cudaMemcpy(device_cities, host_cities, sizeof(host_cities), cudaMemcpyHostToDevice));

    cudaEventRecord(start,0);
    permute_kernel<<<1, NUMCITIES>>>(device_cities, NUMCITIES);
    cudaEventRecord(stop,0);

    GPUerrchk(cudaPeekAtLastError());
    GPUerrchk(cudaDeviceSynchronize());

    cudaEventElapsedTime( &time, start, stop );
    printf("\nTiempo de Ejecucion: %f mSeg\n\n", time);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    cudaFree(device_cities);

    return 0;
}


