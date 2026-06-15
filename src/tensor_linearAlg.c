#include "../include/main.h"
#include <stdlib.h>

tensor *tensorTranspose(const tensor *ten, const int *axes)
{
    int *resultShape = malloc(ten->dimensions * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return NULL;
    }

    // reshape the tensor
    for (int i = 0; i < ten->dimensions; ++i)
    {
        resultShape[i] = ten->shape[axes[i]];
    }

    tensor *result = createTensor(ten->dimensions, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to malloc for result tensor!\n");
        free(resultShape);
        return NULL;
    }
    free(resultShape);

    int *newCoords = malloc(ten->dimensions * sizeof(int));
    if (newCoords == NULL)
    {
        printf("Error: Failed to malloc for newCoords!\n");
        destroyTensor(result);
        return NULL;
    }

    int *oldCoords = malloc(ten->dimensions * sizeof(int));
    if (oldCoords == NULL)
    {
        printf("Error: Failed to malloc for oldCoords!\n");
        free(newCoords);
        destroyTensor(result);
        return NULL;
    }

    for (int i = 0; i < result->size; ++i)
    {
        // locate addresses in the new tensor
        // row-major
        int temp = i;
        for (int d = result->dimensions - 1; d >= 0; --d)
        {
            newCoords[d] = temp % result->shape[d]; // jump this number of steps in dimension d
            temp /= result->shape[d]; // accumulate jumps at dimension d and bring it onto dimension d - 1
        }

        // maps back to the address in the old tensor
        for (int d = 0; d < result->dimensions; ++d)
        {
            oldCoords[axes[d]] = newCoords[d];
        }

        int oldI = 0;
        int multiplier = 1;
        for (int d = result->dimensions - 1; d >= 0; --d)
        {
            oldI += oldCoords[d] * multiplier; // accumulate jumps
            multiplier *= ten->shape[d]; // magnitude of a jump at the next dimension, so-called "weight" of each dimension
        }

        // finally assign to the new tensor's data
        result->data[i] = ten->data[oldI];
    }

    free(newCoords);
    free(oldCoords);
    return result;
}

tensor *tensorBatchedMultiply(const tensor *ten1, const tensor *ten2)
{
    // ten1->shape: {10, 5, 3, 4}
    // to be multiplicable: ten2->shape: {10, 5, 4, int N}
    // whereas: {10, 5}: batched dimensions

    // validation
    if (ten1->dimensions < 2 || ten2->dimensions < 2)
    {
        printf("Error: Tensors' ranks must be at least 2!\n");
        return NULL;
    }
    else if (ten1->dimensions != ten2->dimensions)
    {
        printf("Error: Tensors' dimensions must be the same!\n");
        return NULL;
    }
    int dims = ten1->dimensions;

    // check batch dimensions (0..(dimensions - 3))
    for (int d = 0; d < dims - 2; ++d)
    {
        if (ten1->shape[d] != ten2->shape[d])
        {
            printf("Error: Dissimilar batch dimension!\nten1->shape[%i] != ten2->shape[%i]!\n", d, d);
            return NULL;
        }
    }

    // obtain the actual dimensionality of the will-be-multiplied matrices
    int firstCols = ten1->shape[dims - 1];
    int secondRows = ten2->shape[dims - 2];
    if (secondRows != firstCols)
    {
        printf("Error: matrices not multiplicable!\n");
        return NULL;
    }
    int firstRows = ten1->shape[dims - 2];
    int secondCols = ten2->shape[dims - 1];
    
    // resultShape
    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return NULL;
    }
    for (int i = 0; i < dims - 1; ++i)
    {
        resultShape[i] = ten1->shape[i];
    }
    resultShape[dims - 1] = secondCols;
    
    // TẠO Tensor result với result_shape (Ví dụ result sẽ có shape {10, 5, 3, 6})
    // ĐIỀN SỐ 0 CHO TOÀN BỘ result (Bắt buộc để cộng dồn)
    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        return NULL;
    }
    free(resultShape);

    tensorFillZEROS(result);

    // total numbers of matrices to be multiplied
    int totBatches = 1;
    for (int i = 0; i < dims - 2; ++i)
    {
        totBatches *= ten1->shape[i];
    }

    // // 2. Tính xem 1 ma trận con chiếm bao nhiêu ô nhớ trên RAM 1D
    int sizeFirst = firstRows * firstCols;
    int sizeSecond = secondRows * secondCols;
    int sizeResult = firstRows * secondCols;

    // multiply each pair of matrices
    for (int b = 0; b < totBatches; ++b)
    {
        // offset
        int offsetFirst = b * sizeFirst;
        int offsetSecond = b * sizeSecond;
        int offsetResult = b * sizeResult;

        // matrix multpiplication
        for (int i = 0; i < firstRows; ++i)
        {
            for (int k = 0; k < firstCols; ++k)
            {
                for (int j = 0; j < secondCols; ++j)
                {
                    int indexFirst = offsetFirst + (i * firstCols + k);
                    int indexSecond = offsetSecond + (k * secondCols + j);
                    int indexResult = offsetResult + (i * secondCols + j);
                    result->data[indexResult] += ten1->data[indexFirst] * ten2->data[indexSecond];
                }
            }
        }
    }
    return result;
}

tensor *tensorInverse(const tensor *ten);
