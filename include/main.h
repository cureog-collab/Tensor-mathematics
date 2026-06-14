#ifndef MAIN_H
#define MAIN_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    double *data; // actual data stored
    int *shape; // store the size of each dimension of the tensor
    int dimensions; // dimension of the tensor
    int size; // total components from every dimension
} tensor;

// MEMORY HANDLING =========================================================================================
tensor *createTensor(int dim, const int *shape);
void destroyTensor(tensor *ten);
// =========================================================================================================

// TENSOR OPERATORS ========================================================================================
void tensorFill(tensor *ten, double A);
void tensorFillZEROS(tensor *ten);
void tensorFillONES(tensor *ten);
void tensorRandomUinformUnity(tensor *ten);
void tensorRandomNormal(tensor *ten, double mean, double sigma);

// ---------------------------------------------------------------------------------------------------------

void tensorScale(tensor *ten, double scalar);
bool tensorAdd(tensor *out, const tensor *ten1, const tensor *ten2, bool isAdd);
bool tensorHadamardProduct(tensor *result, const tensor *ten1, const tensor *ten2);
bool tensorRelu(tensor *result, const tensor *ten);
bool tensorSigmoid(tensor *result, const tensor *ten);
bool tensorTanh(tensor *result, const tensor *ten);

// ---------------------------------------------------------------------------------------------------------

bool tensorTranspose(tensor *result, const tensor *ten);
bool tensorInverse(tensor *result, const tensor *ten);
bool tensorMultiply(tensor *result, const tensor *ten1, const tensor *ten2);

// ---------------------------------------------------------------------------------------------------------

double tensorSum(const tensor *ten);
double tensorMean(const tensor *ten);
bool tensorSumByAxis(tensor *result, const tensor *ten, int axis);

// ---------------------------------------------------------------------------------------------------------

bool tensorReshape(tensor *result, const tensor *ten, int *newShape, int *newDim);
bool tensorFlatten(tensor *result, const tensor *ten);
// =========================================================================================================

#endif