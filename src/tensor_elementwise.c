#include "../include/main.h"

bool checkShapeSim(const tensor *ten1, const tensor *ten2);

void tensorScale(tensor *ten, double scalar)
{
    if (isnan(scalar) || isinf(scalar))
    {
        return;
    }

    for (int i = 0; i < ten->size; ++i)
    {
        ten->data[i] *= scalar;
    }
}

tensor *tensorAdd(const tensor *ten1, const tensor *ten2, bool isAdd)
{
    if (!checkShapeSim(ten1, ten2))
    {
        return NULL;
    }

    tensor *result = createTensor(ten1->dimensions, ten1->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }
    if (isAdd)
    {
        for (int i = 0; i < ten1->size; ++i)
        {
            result->data[i] = ten1->data[i] + ten2->data[i];
        }
    }
    else
    {
        for (int i = 0; i < ten1->size; ++i)
        {
            result->data[i] = ten1->data[i] - ten2->data[i];
        }
    }

    return result;
}

tensor *tensorHadamardProduct(const tensor *ten1, const tensor *ten2)
{
    if (!checkShapeSim(ten1, ten2))
    {
        return NULL;
    }
    
    tensor *result = createTensor(ten1->dimensions, ten1->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }
    for (int i = 0; i < ten1->size; ++i)
    {
        result->data[i] = ten1->data[i] * ten2->data[i];
    }

    return result;
}

tensor *tensorRelu(const tensor *ten)
{
    tensor *result = createTensor(ten->dimensions, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }
    for (int i = 0; i < ten->size; ++i)
    {
        result->data[i] = (ten->data[i] > 0) ? ten->data[i] : 0;
    }

    return result;
}

tensor *tensorSigmoid(const tensor *ten)
{
    tensor *result = createTensor(ten->dimensions, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }
    for (int i = 0; i < ten->size; ++i)
    {
        result->data[i] = 1.0 / (1 + exp(-ten->data[i]));
    }

    return result;
}

tensor *tensorTanh(const tensor *ten)
{
    tensor *result = createTensor(ten->dimensions, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }
    for (int i = 0; i < ten->size; ++i)
    {
        result->data[i] = tanh(ten->data[i]);
    }

    return result;
}

// helper:
// check if the two tensors are the same shape
bool checkShapeSim(const tensor *ten1, const tensor *ten2)
{
    if (ten1->dimensions != ten2->dimensions)
    {
        printf("Error: Dimensions of the two tensors are not equal: %i and %i.\n", ten1->dimensions, ten2->dimensions);
        return false;
    }
    if (memcmp(ten1->shape, ten2->shape, ten1->dimensions * sizeof(int)) != 0)
    {
        printf("Error: Shapes of the two tensors are not equal!\n");
        return false;
    }
    return true;
}