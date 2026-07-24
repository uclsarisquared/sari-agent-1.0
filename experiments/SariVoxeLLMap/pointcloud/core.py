from time import time
from dotenv import load_dotenv
import os, sys, logging, trimesh, glob, asyncio, threading
from typing import List
import numpy as np

from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors, _filter_and_downsample, _compute_alignment_transform_first_cam_glTF_center_by_points, _estimate_scene_scale, _add_cameras_to_scene

from .inputs import PointcloudInputs
from .outputs import PointcloudOutputs
from .da3_wrapper import DA3Wrapper

# For ensuring access to DA3 modules in the wrapper
load_dotenv()
da3_path = os.getenv("DA3_PATH")
if da3_path is None:
    raise ValueError("DA3_PATH environment variable not set. Please set it to the path of the Depth-Anything-3 repository.")
sys.path.append(da3_path)  # Adjust the path as needed to access DA3 modules

# Global list to track background plotting threads
_plot_threads: List[threading.Thread] = []
_thread_lock = threading.Lock()

class DynamicPointcloud():

    def __init__(
        self,
        num_frame_overlap: int = 3,
        model_id: str = "DA3-BASE",
        process_res: int = 512,
        save_dir: str = "",
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1000000,
        ensure_thresh_percentile: float = 50.0,
        conf_thresh: float = 0.5,
        delete_old: dict = {
            "inputs": True,  # Whether to delete old saved inputs when saving new ones (to save storage space)
            "outputs": True,  # Whether to delete old saved outputs when saving new
            "pointcloud": False,  # Whether to delete old saved point clouds when saving new ones
            "glb": False,  # Whether to save all GLB exports or just the final one (since they can be large)
        },
    ) -> None:
        
        """
        Initialize the DynamicPointcloud instance with the specified parameters, including the DA3 model configuration, input management, and output management.
        
        :param self: The instance of the DynamicPointcloud class.
        :type self
        :param num_frame_overlap: The number of overlapping frames to use for each window of inference (e.g., 3).
        :type num_frame_overlap: int
        :param model_id: The identifier for the DA3 model variant to use (e.g., "DA3-BASE").
        :type model_id: str
        :param process_res: The processing resolution for depth estimation (e.g., 512).
        :type process_res: int
        :param save_dir: The directory for saving intermediate results and final outputs.
        :type save_dir: str
        :param conf_thresh_percentile: The percentile for adaptive confidence thresholding (e.g., 40.0).
        :type conf_thresh_percentile: float
        :param num_max_points: The maximum number of points after downsampling (e.g., 1,000,000).
        :type num_max_points: int
        :param ensure_thresh_percentile: The percentile for ensuring confidence thresholding when updating the overall prediction (e.g., 50.0).
        :type ensure_thresh_percentile: float
        :param conf_thresh: The minimum confidence threshold for filtering points in the point cloud (e.g., 0.5).
        :type conf_thresh: float
        :param delete_old: A dictionary specifying whether to delete old saved files for each component (inputs, outputs, pointcloud, glb).
        :type delete_old: dict
        """

        self.delete_old = delete_old
        self.save_dir = save_dir
        # Create the save directory if it does not exist
        os.makedirs(self.save_dir, exist_ok=True)

        self.num_max_points = num_max_points

        self.inputs = PointcloudInputs(save_dir=os.path.join(save_dir, "inputs"), num_frame_overlap=num_frame_overlap)  # Instance of PointcloudInputs to manage the inputs for dynamic point cloud prediction
        self.da3_wrapper = DA3Wrapper(model_id, process_res, conf_thresh_percentile, num_max_points)  # Instance of DA3Wrapper to handle model inference
        self.outputs = PointcloudOutputs(os.path.join(save_dir, "outputs"), conf_thresh_percentile, ensure_thresh_percentile, conf_thresh)  # Instance of PointcloudOutputs to manage the outputs for dynamic point cloud prediction

        # Final point cloud
        self.points = np.array([]).reshape(0, 3)    # np.ndarray of shape (X, 3) containing the current set of 3D points in world coordinates after the latest update
        self.colors = np.array([]).reshape(0, 3)    # np.ndarray of shape (X, 3) containing the RGB color values for each point in the current point cloud

        self.logger = logging.getLogger(__name__)

        # For parallel operation for receiving and processing frames
        self.frame_queue = asyncio.Queue()  # Holds tuples: (image_path, extrinsics, intrinsics)

    def __call__(self, new_frame_image_path: str, new_frame_extrinsics: np.ndarray, new_frame_intrinsics: np.ndarray) -> dict | None:
        """
        Update the dynamic point cloud with a new frame.
        This method takes the new frame's image path, extrinsics, and intrinsics,
        runs the DA3 model inference on the current window of frames (including the new frame),
        updates the overall prediction with confidence thresholding and alignment,
        and then extracts and returns the updated point cloud.

        :param self: The instance of the DynamicPointcloud class.
        :type self
        :param new_frame_image_path: The file path to the new frame's image.
        :type new_frame_image_path: str
        :param new_frame_extrinsics: The extrinsics array for the new frame, representing the camera's pose in world coordinates.
        :type new_frame_extrinsics: np.ndarray
        :param new_frame_intrinsics: The intrinsics array for the new frame, representing the camera's internal parameters.
        :type new_frame_intrinsics: np.ndarray | None
        :return: A numpy array of 3D points representing the updated point cloud after integrating the new frame.
        :rtype: np.ndarray
        """
        
        # 1) Append the new input data to the inputs manager
        start_ = time()
        window_image_paths, window_extrinsics, window_intrinsics = self.inputs(new_frame_image_path, new_frame_extrinsics, new_frame_intrinsics)  
        self.logger.info(f"Inputs updated. Time: {time() - start_:.2f} seconds")

        # 2) Run the DA3 model inference using the current window with the new frame
        if len(self.inputs) >= 3:  # Only run inference if we have at least 3 frames (since DA3 relies on multi-frame input)

            # Wait for ongoing prediction to finish if necessary (this can be implemented with a lock or flag if running in a multi-threaded environment)
            start_ = time()
            prediction = self.da3_wrapper(window_image_paths, window_extrinsics, window_intrinsics)  
            self.logger.info(f"DA3 inference completed. Time: {time() - start_:.2f} seconds")

            # 4) Update the outputs manager with the new prediction, which handles
            # confidence thresholding and alignment of the new prediction with the existing overall prediction
            start_ = time()
            self.outputs(prediction.depth, prediction.conf, prediction.processed_images, prediction.intrinsics)
            self.logger.info(f"Outputs updated. Time: {time() - start_:.2f} seconds")

            # 5) Convert the updated overall prediction into a point cloud, applying confidence thresholding, back-projection to world coordinates, alignment, and downsampling
            start_ = time()
            points = self.update_points()
            self.logger.info(f"Point cloud updated: {points.shape[0]} points. Time: {time() - start_:.2f} seconds")

            # 6) Return all data that might be needed either for downstream processing or saves
            return {
                'points': self.points,
                # For GLB export
                'extrinsics': self.inputs.extrinsics,
                'intrinsics': self.outputs.intrinsics,
                'depth_shape': self.outputs.depth.shape,
                'colors': self.colors,
            }
    
    def __repr__(self) -> str:

        """
        Return a string representation of the DynamicPointcloud instance, including key parameters from the inputs manager, DA3 wrapper, and outputs manager for easy debugging and visualization of the current configuration.
        :param self: The instance of the DynamicPointcloud class.
        :type self
        :return: A string representation of the DynamicPointcloud instance with key parameters.
        :rtype: str
        """

        return f"DynamicPointcloud(num_frame_overlap={self.inputs.num_frame_overlap}, model_id='{self.da3_wrapper.model_id}', process_res={self.da3_wrapper.process_res}, conf_thresh_percentile={self.da3_wrapper.conf_thresh_percentile}, num_max_points={self.da3_wrapper.num_max_points}, ensure_thresh_percentile={self.outputs.ensure_thresh_percentile}, conf_thresh={self.outputs.conf_thresh})"

    def update_points(self) -> np.ndarray:
        
        """
        Extract 3D points from a DA3 prediction object, applying confidence thresholding, back-projection to world coordinates, alignment, and downsampling.
        Modified from depth_anything_3.utils.export.glb.export_to_glb

        :param self: The instance of the DynamicPointcloud class.
        :type self
        :return: A numpy array of 3D points extracted from the prediction, formatted as (N, 3) array of (x, y, z) coordinates.
        :rtype: np.ndarray
        """

        start = time()

        # 1) Back-project to world coordinates and get colors (world frame)
        self.points, self.colors = _depths_to_world_points_with_colors(
            self.outputs.depth,
            self.outputs.intrinsics,
            self.inputs.extrinsics,  # w2c
            self.outputs.processed_images,
            self.outputs.conf,
            self.outputs.dynamic_conf_thresh,
        )

        # TODO: Review whether filtering is needed here for redundant points, affecting density
        # Potential problem: will affect latency, since it might have to search through all existing camera poses

        # Maybe, look if there are frames with almost similar poses and then randomly drop points
        # OR, compute for overlapping viewpoints?
        # Might have to use the position and rotation vectors if so
        # Pseudocode
        # if ()

        # ORRR identify the points from the new window AND THEN do separate processing for them?

        # or retain na lang??

        return self.points