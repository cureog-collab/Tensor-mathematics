#include "../include/main.h"
#include <string.h>

double tensorSum(const tensor *ten)
{
    if (ten == NULL)
    {
        printf("Error: Cannot perform tensorSum on NULL!\n");
        return NAN;
    }

    double sum = 0;
    for (int i = 0; i < ten->size; ++i)
    {
        sum += ten->data[i];
    }

    return sum;
}

double tensorMean(const tensor *ten)
{
    double sum = tensorSum(ten);
    if (isnan(sum))
    {
        return NAN;
    }

    return (sum / ten->size);
}

tensor *tensorSumByAxis(const tensor *ten, int axis); // TODO

// tensor *tensorMaxByAxis(const tensor *ten, int axis)
// {
//     if (ten == NULL)
//     {
//         printf("Error: Input tensor is NULL!\n");
//         return NULL;
//     }

//     if (axis < 0 || axis >= ten->dimensions)
//     {
//         printf("Error: Axis %i is not a valid axis!\n", axis);
//         return NULL;
//     }

//     int dims = ten->dimensions;

//     int *resultShape = malloc(dims * sizeof(int));
//     if (resultShape == NULL)
//     {
//         printf("Error: Failed to malloc for resultShape!\n");
//         return NULL;
//     }
//     memcpy(resultShape, ten->shape, dims * sizeof(int));
//     resultShape[axis] = 1;

//     tensor *result = createTensor(dims, resultShape);
//     if (result == NULL)
//     {
//         printf("Error: Failed to create result tensor!\n");
//         free(resultShape);
//         return NULL;
//     }
//     free(resultShape);
// }

tensor *tensorArgmax(const tensor *ten, int axis);    // TODO