"""detector object for running on single images"""

from torchvision import transforms
import numpy as np
import torch

from .. import bbox_utils


class TorchDetector:
    """object detector to be run on single images

    Args:
        model (nn.Module):
        decoder (Decoder):
        input_dim (int): image resized to (input_dim, input_dim) before being
            fed to model.
        max_outputs (int):
        nms_iou_thresh (float):

    Attributes:
        model (nn.Module): see Args.
        decoder (Decoder): see Args.
        max_outputs (int): see Args.
        nms_iou_thresh (float): see Args.
        img_to_x (Transform): resizes image and converts it from
            PIL.Image -> Tensor.
        softmax (nn.Module): model outputs are not softmaxed
            (done this way so softmax can be combined with loss function for
            numerical stability). Detector is expected to perform softmax as
            postprocessing step
    """
    def __init__(
            self, model, decoder, input_dim, max_outputs, nms_iou_thresh
    ):
        self.model = model
        self.decoder = decoder
        self.max_outputs = max_outputs
        self.nms_iou_thresh = nms_iou_thresh

        self.img_to_x = transforms.Compose([
            transforms.Resize((input_dim, input_dim)),
            transforms.ToTensor()
        ])
        self.softmax = torch.nn.Softmax(dim=2)

    @torch.no_grad()
    def __call__(self, img):
        """run object detector on img

        Args:
            img (Image): PIL image to run predictions on
        Returns:
            classes (array): predicted class IDs
            boxes (array): predicted bounding boxes; ijhw, fractional coords
        """
        ### prepare for model
        img = self.img_to_x(img)
        img.unsqueeze_(0)  # insert batch dimension

        ### model predictions
        c_hat, b_hat = self.model(img.cuda())
        c_hat = self.softmax(c_hat)
        c_hat = self._process_model_output(c_hat)
        b_hat = self._process_model_output(b_hat)

        ### decode model outputs
        confs, classes, boxes = self.decoder(c_hat, b_hat)

        ### truncating outputs
        conf_inds = np.argsort(confs)[-self.max_outputs:]
        confs = confs[conf_inds]
        classes = classes[conf_inds]
        boxes = boxes[conf_inds, :]

        ### non-max suppression
        nms_mask = bbox_utils.get_nms_mask(confs, boxes)
        confs = confs[nms_mask]
        classes = classes[nms_mask]
        boxes = boxes[nms_mask, :]

        return confs, classes, boxes

    @staticmethod
    def _process_model_output(tensor):
        """get rid of batch dimension and convert to numpy array"""
        return tensor.squeeze(0).cpu().numpy()

    def load_state_dict(self, state_dict):
        """update detector model with new weights"""
        self.model.load_state_dict(state_dict)
