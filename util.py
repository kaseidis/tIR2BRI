import torch
import torchvision.transforms as transforms
from skimage import img_as_uint



def imageToInput(img):
    """"Convert grayscale scikit image to input tensor

    Args:
        img (ndarray): single channel uint8 scikit image (W,H)

    Returns:
        tensor: tensor of (1,1,W,H)
    """
    convert_tensor = transforms.ToTensor()
    input = convert_tensor(img)
    input = input.reshape((1, 1, img.shape[0], img.shape[1])) * 2 - 1
    return input

def inputToImage(tensor):
    """"Convert grayscale scikit image to input tensor

    Args:
        tensor (tensor): tensor of (1,1,W,H)

    Returns:
        ndarray: uint8 scikit image (W,H)
    """
    result = torch.reshape(tensor,tensor.shape[2:4]).cpu().detach().numpy()
    result = (result + 1) / 2
    result = img_as_uint(result)
    return result
