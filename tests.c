#include <stdio.h>
#include <math.h>
#include "base.c"

#define EPSILON 1e-5

int main(int argc, char **argv) {
    temp_arena = arena_new(1024 * Kb);

    // matmul tests
    float a[2][3] = {
        {1, 2, 3},
        {4, 5, 6}
    };
    float b[3][2] = {
        {1, 2},
        {3, 4},
        {5, 6}
    };
    // Manual calculation for expected result:
    // [1 2 3]   [1 2]   [1*1+2*3+3*5  1*2+2*4+3*6]   [1+6+15  2+8+18]   [22 28]
    // [4 5 6] x [3 4] = [4*1+5*3+6*5  4*2+5*4+6*6] = [4+15+30 8+20+36] = [49 64]
    //           [5 6]
    // So expected is:
    // [22 28]
    // [49 64]
    Tensor a_tensor = (Tensor){(float*)a, ten_shape(2, 3)};
    Tensor b_tensor = (Tensor){(float*)b, ten_shape(3, 2)};
    Tensor c_tensor = ten_matmul(temp_arena, a_tensor, b_tensor);

    float expected[2][2] = {
        {22, 28},
        {49, 64}
    };

    bool passed = true;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            float actual = c_tensor.data[i * 2 + j];
            float exp = expected[i][j];
            printf("c[%d][%d] = %f (expected %f) ", i, j, actual, exp);
            if(fabs(actual - exp) > EPSILON) {
                printf("(FAILED at row %d, col %d: got %f, expected %f) ", i, j, actual, exp);
                passed = false;
            }
        }
        printf("\n");
    }

    if(passed) {
        printf("Matmul tests passed\n");
    } else {
        printf("Matmul tests failed\n");
    }

    return 0;
}