#define FORCE_IMPORT_ARRAY
#include "prediction_filtering.hpp"

#include "pybind11/pybind11.h"
#include "xtensor/xview.hpp"
#include "xtensor/xsort.hpp"

#include "bbox_utils.hpp"


template<typename Tensor>
Tensor PredictionFilter::applyMask(
    const BoolTensor1D& mask, const Tensor& toMask
)
{
    return xt::view(toMask, xt::keep(mask));
}


PredictionFilterPipeline::PredictionFilterPipeline(
    const PredictionFilter& filterLayers, ...
)
{
    ;
}


std::tuple<FloatTensor1D, IntTensor1D, FloatTensor2D>
PredictionFilterPipeline::filter(
    const FloatTensor1D& confs,
    const IntTensor1D& classes,
    const FloatTensor2D& boxes
)
{
    ;
}


std::tuple<FloatTensor1D, IntTensor1D, FloatTensor2D>
ConfidenceFilter::filter(
    const FloatTensor1D& confs,
    const IntTensor1D& classes,
    const FloatTensor2D& boxes
)
{
    const BoolTensor1D mask = confs > ConfidenceFilter::confThresh;

    const FloatTensor1D maskedConfs = PredictionFilter::applyMask(mask, confs);
    const IntTensor1D maskedClasses = PredictionFilter::applyMask(mask, classes);
    const FloatTensor2D maskedBoxes = PredictionFilter::applyMask(mask, boxes);
    return std::make_tuple(maskedConfs, maskedClasses, maskedBoxes);
}


std::tuple<FloatTensor1D, IntTensor1D, FloatTensor2D>
NMSFilter::filter(
    const FloatTensor1D& confs,
    const IntTensor1D& classes,
    const FloatTensor2D& boxes
)
{
    std::vector<FloatTensor1D> confsVect {xt::empty<float>({0})};
    std::vector<IntTensor1D> classesVect {xt::empty<int>({0})};
    std::vector<FloatTensor2D> boxesVect {xt::empty<float>({0, 4})};

    for (int clsID : xt::unique(classes)) {
        BoolTensor1D clsMask = xt::equal(classes, clsID);
        FloatTensor1D clsConfs = PredictionFilter::applyMask(clsMask, confs);
        FloatTensor2D clsBoxes = PredictionFilter::applyMask(clsMask, boxes);

        BoolTensor1D nmsMask = getNMSMask(clsConfs, clsClasses, clsBoxes);
        FloatTensor1D nmsConfs = PredictionFilter::applyMask(nmsMask, clsConfs);
        FloatTensor2D nmsBoxes = PredictionFilter::applyMask(nmsMask, clsBoxes)

        IntTensor1D nmsClasses = xt::zeros<int>(nmsConfs.shape()[0]) * clsID;

        confsVect.push_back(nmsConfs);
        classesVect.push_back(nmsClasses);
        boxesVect.push_back(nmsBoxes);
    }
    filteredConfs = xt::concatenate(xt::)
}


PYBIND11_MODULE(encoding, m)
{
    xt::import_numpyclsMask);
    m.doc() = "prediclsMasktion filter classes";

    pybind11::class<ConfidenceFilter>(m, "ConfidenceFilter")
        .def(pybind11::init<float>())
        .def("__call__", &ConfidenceFilter::filter)
        .def_readonlclsMask("conf_thresh", &ConfidenceFilter::confThresh);
}