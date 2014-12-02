#include <stdio.h>

struct City {
    int x, y;
    char name;
};

inline void GPUassert(cudaError_t code, char * file, int line, bool Abort=true)
{
    if (code != 0) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
        if (Abort) exit(code);
    }       
}

#define GPUerrchk(ans) { GPUassert((ans), __FILE__, __LINE__); }

#define NUMCITIES 9
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

__device__ void print_path(struct City *path){
    for(int i = 0; i < NUMCITIES; i++){
        printf("%c>", path[i].name);
    }
    printf("\n");
}

__device__ void format_path(struct City *path, char *str){
    for(int i = 0; i < NUMCITIES; i++){
        *str = path[i].name;        
        str++;
        *str = '>';
        str++;
    }
    str--; *str = 0;
}

__device__ void permutations_kernel(struct City *a, char **paths, double *distances, int i, int length, int tid, int *count) { 

    if (length == i){
        long distance = total_distance(a);
        //format_path(a, paths[count[0]]);
        count[0] = count[0] + 1;
    } else {
        for (int j = i; j < length; j++) {
            swap(a, i, j);
            // CUDA
            // permutations(a, i+1, length, tid, count);
            permutations_kernel(a, paths, distances, i+1, length, tid, count);
            swap(a, i, j);
        }
    }
}


__global__ void permute_kernel(struct City *dev_cities, char **paths, double *distances, int size) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int count[1]; 
    count[0] = 0;

    struct City local_array[NUMCITIES];

    for (int i=0; i<size; i++){
        local_array[i] = dev_cities[i];  
    } 

    //swap(local_array + threadIdx.x, local_array);
    //swap(local_array, threadIdx.x, 0);
    permutations_kernel(local_array, paths, distances, 0, NUMCITIES, tid, count);

}

long factorial(int i) {
    long result = 1;
    while(i > 0) {		
        result *= i;
	i--;
    }
    return result;
}

int main(){

    struct City host_cities[NUMCITIES];
    for(int c = 0; c < NUMCITIES; c++){
        host_cities[c].name = 'A' + c;
        host_cities[c].x = rand() % 20 + 5;
        host_cities[c].y = rand() % 20 + 5;
    }

    //char host_paths [ factorial(NUMCITIES) ][ NUMCITIES*NUMCITIES ];
    char host_paths [0][0]; 
    char **device_paths;

    //double host_distances[ factorial(NUMCITIES) ];
    double host_distances[0];
    double *device_distances;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    struct City *device_cities; 
    cudaMalloc((void**) &device_cities, sizeof(host_cities));
    //cudaMalloc((void**) &device_paths, sizeof(host_paths));
    cudaMalloc((void**) &device_paths, sizeof(char) * NUMCITIES * NUMCITIES * factorial(NUMCITIES));
    cudaMalloc((void**) &device_distances, sizeof(host_distances));

    GPUerrchk(cudaMemcpy(device_distances, host_distances, sizeof(host_distances), cudaMemcpyHostToDevice));
    GPUerrchk(cudaMemcpy(device_cities, host_cities, sizeof(host_cities), cudaMemcpyHostToDevice));
    //GPUerrchk(cudaMemcpy(device_paths, host_paths, sizeof(host_paths), cudaMemcpyHostToDevice));
    GPUerrchk(cudaMemcpy(device_paths, host_paths, sizeof(char) * NUMCITIES * NUMCITIES * factorial(NUMCITIES), cudaMemcpyHostToDevice));   

    cudaEventRecord(start,0);
    permute_kernel<<<1, NUMCITIES>>>(device_cities, device_paths, device_distances, NUMCITIES);
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


