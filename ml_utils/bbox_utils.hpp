#include "typedefs.hpp"


FloatTensor2D ijhwToIjij(const FloatTensor2D& boxes);

inline FloatTensor1D computeIjijArea(const FloatTensor2D& boxesIjij);

FloatTensor2D computeIOUs(
    const FloatTensor2D& rowBoxes, const FloatTensor2D& colBoxes
);

BoolTensor1D getNMSMask(
    const FloatTensor1D& confs,
    const FloatTensor2D& boxes,
    const float iouThresh
);