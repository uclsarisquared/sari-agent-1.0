import torch, sys, os, logging
import numpy as np

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction

sys.path.append("C:\\Users\\Kat Gados\\Documents\\Thesis\\Code\\2526\\agent\\Depth-Anything-3\\src\\depth_anything_3")  # Adjust the path as needed to access DA3 modules

logger = logging.getLogger(__name__)

class DA3Wrapper:

    def __init__(self, model_id: str = "DA3-BASE", process_res: int = 512, conf_thresh_percentile: float = 40.0, num_max_points: int = 1000000):
        """
        Initialize the DA3Wrapper instance with the specified parameters and load the DA3 model.

        :param model_id: The identifier for the DA3 model variant to use (e.g., "DA3-BASE").
        :type model_id: str
        :param process_res: The processing resolution for depth estimation (e.g., 512).
        :type process_res: int
        :param conf_thresh_percentile: The percentile for adaptive confidence thresholding (e.g., 40.0).
        :type conf_thresh_percentile: float
        :param num_max_points: The maximum number of points after downsampling (e.g., 1,000,000).
        :type num_max_points: int
        """

        self.model_id = model_id
        self.process_res = process_res
        self.conf_thresh_percentile = conf_thresh_percentile
        self.num_max_points = num_max_points
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DepthAnything3.from_pretrained(f"depth-anything/{model_id}")
        self.model = model.to(device=device)

    def __call__(self, image_paths: list[str], extrinsics: np.ndarray, intrinsics: np.ndarray) -> Prediction:    
        
        """
        Run inference on a window of frames based on the configured number of frame overlaps and return the prediction for that window.
        
        :param self: The instance of the DA3Wrapper class.
        :type self
        :param image_paths: The list of image file paths for the window of frames to be used for inference.
        :type image_paths: list[str]
        :param extrinsics: The extrinsics array for the window of frames to be used for inference.
        :type extrinsics: np.ndarray
        :param intrinsics: The intrinsics array for the window of frames to be used for inference.
        :type intrinsics: np.ndarray
        :param export_dir: The export directory for the GLB files.
        :type export_dir: str
        :return: The prediction object containing depth maps and camera parameters for the selected window of frames.
        :rtype: Prediction
        """    

        # Checks
        assert len(image_paths) == extrinsics.shape[0] == intrinsics.shape[0], "The number of image paths, extrinsics, and intrinsics must be the same."
        assert extrinsics.shape == (len(image_paths), 4, 4), "Extrinsics must have shape (N, 4, 4), got {}".format(extrinsics.shape)
        assert intrinsics.shape == (len(image_paths), 3, 3), "Intrinsics must have shape (N, 3, 3), got {}".format(intrinsics.shape)

        # Wait until all screenshots have been saved
        while not all(os.path.exists(path) for path in image_paths):
            pass

        return self.model.inference(
            image_paths,
            extrinsics,
            intrinsics,
            process_res=self.process_res
        )
    
    def __repr__(self) -> str:
        """
        Get a string representation of the DA3Wrapper instance, including the model ID and inference parameters.

        :return: A string representation of the DA3Wrapper instance.
        :rtype: str
        """

        return f"DA3Wrapper(model_id={self.model_id}, process_res={self.process_res}, conf_thresh_percentile={self.conf_thresh_percentile}, num_max_points={self.num_max_points})"