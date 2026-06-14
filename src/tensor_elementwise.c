#include "../include/main.h"

void tensorScale(tensor *ten, double scalar);
bool tensorAdd(tensor *out, const tensor *ten1, const tensor *ten2, bool isAdd);
bool tensorHadamardProduct(tensor *result, const tensor *ten1, const tensor *ten2);
bool tensorRelu(tensor *result, const tensor *ten);
bool tensorSigmoid(tensor *result, const tensor *ten);
bool tensorTanh(tensor *result, const tensor *ten);