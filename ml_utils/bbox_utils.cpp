#define FORCE_IMPORT_ARRAY  // required before any includes from xtensor-python
#include "bbox_utils.hpp"

#include <unordered_set>
#include <vector>

#include "pybind11/pybind11.h"
#include "xtensor/xview.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xio.hpp"


FloatTensor2D ijhwToIjij(const FloatTensor2D& boxes)
{
    // ijhw values are expected to be in range(0, 1)

    const FloatTensor2D ij = xt::view(boxes, xt::all(), xt::range(0, 2));
    const FloatTensor2D hw_rad = xt::view(boxes, xt::all(), xt::range(2, 4)) / 2;

    FloatTensor2D boxesIjij = xt::concatenate(xt::xtuple(ij-hw_rad, ij+hw_rad), 1);
    boxesIjij = xt::clip(boxesIjij, 0, 1);  // enforce bounds

    return boxesIjij;
}


inline FloatTensor1D computeIjijArea(const FloatTensor2D& boxesIjij)
{
    const FloatTensor2D boxesHW = xt::maximum(
        xt::view(boxesIjij, xt::all(), xt::range(2, 4)) -
        xt::view(boxesIjij, xt::all(), xt::range(0, 2)),
        0
    );
    return xt::prod(boxesHW, {1});
}


FloatTensor2D computeIOUs(
    const FloatTensor2D& rowBoxes, const FloatTensor2D& colBoxes
)
{
    // IOUs computed for each bbox pair (a, b) in the cartesian
    // product of A and B

    const FloatTensor2D& rowBoxesIjij = ijhwToIjij(rowBoxes);
    const FloatTensor2D& colBoxesIjij = ijhwToIjij(colBoxes);

    //// compute intersection
    // compute top left and bottom right corner of each intersection(a, b)
    const FloatTensor3D abInterTL = xt::maximum(
        xt::view(rowBoxesIjij, xt::all(), xt::newaxis(), xt::range(0, 2)),
        xt::view(colBoxesIjij, xt::all(), xt::range(0, 2))
    ); // (|rowBoxes|, |colBoxes|, 2)
    const FloatTensor3D abInterBR = xt::minimum(
        xt::view(rowBoxesIjij, xt::all(), xt::newaxis(), xt::range(2, 4)),
        xt::view(colBoxesIjij, xt::all(), xt::range(2, 4))
    ); // (|rowBoxes|, |colBoxes|, 2)
    // compute H, W of each intersection(a, b)
    const FloatTensor3D abInterHW = xt::maximum(abInterBR - abInterTL, 0);
    // compute area of each intersection(a, b):
    // has shape (|rowBoxes|, |colBoxes|)
    const FloatTensor2D abIntersection = xt::prod(abInterHW, {2});

    //// compute union
    // compute area for each bbox in rowBoxes and colBoxes
    const FloatTensor1D rowAreas = computeIjijArea(rowBoxesIjij);
    const FloatTensor1D colAreas = computeIjijArea(colBoxesIjij);
    // compute aub = a + b - anb for (a, b) in AxB
    const FloatTensor2D abUnion = xt::view(
        rowAreas, xt::all(), xt::newaxis()
    ) + colAreas - abIntersection;

    return abIntersection / abUnion;
}


BoolTensor1D getNMSMask(
    const FloatTensor1D& confs,
    const FloatTensor2D& boxes,
    const float iouThresh
)
{
    IntTensor1D sortedInds = xt::flip(xt::argsort(confs), 0);
    FloatTensor2D ious = computeIOUs(boxes, boxes);
    std::unordered_set<int> keptInds(sortedInds.begin(), sortedInds.end());

    for (int i : sortedInds) {
        if (!keptInds.count(i))  // boxes[i] has already been removed
            continue;

        // suppress_inds = {j : iou(boxes[i], boxes[j] > iou_thresh, j != i)}
        IntTensor1D overlapInds = xt::flatten_indices(
            xt::argwhere(xt::view(ious, i, xt::all()) > iouThresh)
        );
        std::unordered_set<int> indsToSuppress(
            overlapInds.begin(), overlapInds.end()
        );
        indsToSuppress.erase(i);  // dont remove self from keptInds

        for (int j : indsToSuppress)
            keptInds.erase(j);  // TODO - in place set subtraction?
    }
    std::vector<int> keptIndsVect(keptInds.begin(), keptInds.end());
    BoolTensor1D mask = xt::zeros<bool>({confs.shape()[0]});
    xt::view(mask, xt::keep(keptIndsVect)) = true;

    return mask;
}


PYBIND11_MODULE(bbox_utils, m)
{
    xt::import_numpy();
    m.doc() = "bounding box utilities, implemented in C++";
    m.def(
        "ijhw_to_ijij", &ijhwToIjij,
        "convert ijhw bounding boxes to ijij.\n"
        "\n"
        "Args:\n"
        "   bboxes (ndarray): (|bboxes|, 4); bounding boxes to convert. "
        "must be in ijhw format, and have fractional values (in [0, 1])\n"
        "\nm"
        "Returns:\n"
        "   ndarray: (|bboxes|, 4); converted ijij bboxes, with values clipped "
        "to fall in [0, 1]\n"
    );
    m.def(
        "compute_ious", &computeIOUs,
        "compute IOUs of all bounding boxes in cartesian product of "
        "rowBoxes, colBoxes.\n"
        "\n"
        "Args:\n"
        "   colBoxes (ndarray): (|colBoxes|, 4); first set of bounding boxes\n"
        "   rowBoxes (ndarray): (|rowBoxes|, 4); second set of bounding boxes\n"
        "\n"
        "Returns:\n"
        "   ndarray: (|colBoxes|, |rowBoxes|); ious[a, b] = IoU(a, b)\n"
    );
    m.def(
        "get_nms_mask", &getNMSMask,
        "for each set of bboxes with ious above iou_thresh, eliminates all but "
        "highest confidence bbox.\n"
        "\n"
        "Args:\n"
        "   confs (ndarray): (N,); confidence for each box prediction.\n"
        "   bboxes (ndarray): (N, 4); unsuppressed box predictions.\n"
        "   iou_thresh (float): minimum iou between two boxes to be considered "
        "an overlap.\n"
        "\n"
        "Returns:\n"
        "   ndarray: (N,): true where prediction should be kept.\n"
    );
}
