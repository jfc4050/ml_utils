"""detector object for running on single images"""

from torchvision import transforms
import torch


class TorchDetector:
    """object detector to be run on single images

    Args:
        model (nn.Module):
        decoder (Decoder):
        pred_filter (PredictionFilter): used to filter model outputs.
        input_dim (int): image resized to (input_dim, input_dim) before being
            fed to model.

    Attributes:
        model (nn.Module): see Args.
        decoder (Decoder): see Args.
        pred_filter (PredictionFilter): see Args.
        img_to_x (Transform): resizes image and converts it from
            PIL.Image -> Tensor.
        softmax (nn.Module): model outputs are not softmaxed
            (done this way so softmax can be combined with loss function for
            numerical stability). Detector is expected to perform softmax as
            postprocessing step
    """
    def __init__(
            self,
            model,
            decoder,
            input_dim,
            pred_filter=None
    ):
        self.model = model
        self.decoder = decoder
        self.pred_filter = pred_filter

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

        ### filter predictions
        if self.pred_filter is not None:
            confs, classes, boxes = self.pred_filter(confs, classes, boxes)

        return confs, classes, boxes

    @staticmethod
    def _process_model_output(tensor):
        """get rid of batch dimension and convert to numpy array"""
        return tensor.squeeze(0).cpu().numpy()

    def load_state_dict(self, state_dict):
        """update detector model with new weights"""
        self.model.load_state_dict(state_dict)
