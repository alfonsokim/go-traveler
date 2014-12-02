#include <stdio.h>

void printArray(int *, int);

void swap (int *x, int *y){
    int temp;
    temp = *x;
    *x = *y;
    *y = temp;
}

void permutations(int *a, int i, int length) { 

    if (length == i){
        int distance = 0;
        for (int j = 0; j < length; j++) {
            distance += a[j];
        }
        a[length + 1] = distance;
        return;
    }
    int j = i;
    for (int j = i; j <= length; j++) {
        swap((a+i), (a+j));
        // CUDA
        // permutations(a, i+1, length, tid, count);
        permutations(a, i+1, length);
        swap((a+i), (a+j)); 
    }
}

void printArray(int *a, int lenght){
    for(int i = 0; i < lenght; i++){
        printf("%i, ", a[i] );
    }
    //printf("\n");
}

int main(){

    int a[4][5] = {  
        { 20, 5, 13, 9, 0 },
        { 16, 19, 11, 7, 0 },
        { 13, 13, 8, 11, 0 },
        { 5, 10, 5, 6, 0 },
    };

    int test[5] = { 20, 5, 13, 9, 0 };

    permutations(test, 0, 4);
    printArray(test, 5);
    printf("\n");

    return 0;

}  