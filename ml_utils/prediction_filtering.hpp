#include <tuple>

#include "typedefs.hpp"


class PredictionFilter
{
public:
    virtual std::tuple<FloatTensor1D, IntTensor1D, FloatTensor2D> filter(
        const FloatTensor1D& confs,
        const IntTensor1D& classes,
        const FloatTensor2D& boxes
    ) = 0;

protected:
    template<typename Tensor>
    Tensor applyMask(const BoolTensor1D& mask, const Tensor& toMask);
};


class PredictionFilterPipeline : private PredictionFilter
{
public:
    PredictionFilterPipeline(const PredictionFilter& filterLayers, ...);
    std::tuple<FloatTensor1D, IntTensor1D, FloatTensor2D> filter(
        const FloatTensor1D& confs,
        const IntTensor1D& classes,
        const FloatTensor2D& boxes
    );
};


class ConfidenceFilter : private PredictionFilter
{
public:
    const float confThresh;

    ConfidenceFilter(float confThresh) : confThresh(confThresh) {};
    std::tuple<FloatTensor1D, IntTensor1D, FloatTensor2D> filter(
        const FloatTensor1D& confs,
        const IntTensor1D& classes,
        const FloatTensor2D& boxes
    );
};


class NMSFilter : private PredictionFilter
{
public:
    const float iouThresh;

    NMSFilter(float iouThresh) : iouThresh(iouThresh) {};
    std::tuple<FloatTensor1D, IntTensor1D, FloatTensor2D> filter(
        const FloatTensor1D& confs,
        const IntTensor1D& classes,
        const FloatTensor2D& boxes
    );
};
