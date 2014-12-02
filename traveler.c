#include <stdio.h>
//#include <time.h>
#include <stdlib.h>

#define NUMCITIES 4

struct City {
    int x, y;
};

#define CITYSIZE sizeof(struct City)
// #define CITYSIZE 1

void printArray(int *, int);

void swap (struct City *x, struct City *y){
    struct City temp;
    temp = *x;
    *x = *y;
    *y = temp;
}

void swapCity (struct City *a, int x, int y){
    struct City temp;
    temp = a[x];
    a[x] = a[y];
    a[y] = temp;
}

void permutations(struct City *a, int i, int length) { 

    if (length == i){
        printf("====================================\n");
        for(int c = 0; c < length; c++){
            //printf("[x=%i, y=%i]\n", a[c].x, a[c].y);
        }
    } else {
        for (int j = i; j < length; j++) {
            //swap((a + (i * CITYSIZE)), (a + (j * CITYSIZE)));
            swapCity(a, i, j);
            // CUDA
            // permutations(a, i+1, length, tid, count);
            permutations(a, i+1, length);
            //swap((a + (i * CITYSIZE)), (a + (j * CITYSIZE))); 
            swapCity(a, i, j);
        }
    }
}

void printArray(int *a, int lenght){
    for(int i = 0; i < lenght; i++){
        printf("%i, ", a[i] );
    }
}

void printCity(struct City c){
    printf("[x=%i, y=%i]\n", c.x, c.y);
}

int main(){

    //srand(time(NULL));
    struct City cities[NUMCITIES];
    for(int c = 0; c < NUMCITIES; c++){
        cities[c].x = rand() % 20 + 5;
        cities[c].y = rand() % 20 + 5;
        printCity(cities[c]);
    }

    permutations(&cities, 0, NUMCITIES);

    return 0;

}  