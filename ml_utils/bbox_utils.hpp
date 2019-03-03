#include "xtensor-python/pytensor.hpp"

typedef xt::pytensor<bool, 1> BoolTensor1D;
typedef xt::pytensor<int, 1> IntTensor1D;
typedef xt::pytensor<float, 1> FloatTensor1D;
typedef xt::pytensor<float, 2> FloatTensor2D;
typedef xt::pytensor<float, 3> FloatTensor3D;


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