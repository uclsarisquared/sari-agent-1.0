import os, glob
import numpy as np
import logging

class PointcloudOutputs:

    def __init__(self, save_dir: str, conf_thresh_percentile: float, ensure_thresh_percentile: float, conf_thresh: float):
        """
        Initialize the PointcloudOutputs class to manage the outputs for dynamic point cloud prediction.

        The depth, confidence, and processed_images attributes are initialized as empty numpy arrays and will be updated with each new prediction.
        :param save_dir: The directory where outputs should be saved.
        :type save_dir: str
        :param conf_thresh_percentile: The percentile to use for computing the confidence threshold for filtering points in the point cloud.
        :type conf_thresh_percentile: float
        :param ensure_thresh_percentile: The percentile to use for ensuring that the confidence threshold is not too high, which could result in no points being included in the point cloud.
        :type ensure_thresh_percentile: float
        :param conf_thresh: The initial confidence threshold to use for filtering points in the point cloud, which will be updated based on the confidence maps of the predictions.
        :type conf_thresh: float
        """
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.conf_thresh_percentile = conf_thresh_percentile
        self.ensure_thresh_percentile = ensure_thresh_percentile
        self.conf_thresh = conf_thresh

        self.depth = np.empty((0, 0, 0))  # Array to store depth maps for each frame, initialized as empty
        self.conf = np.empty((0, 0, 0))  # Array to store confidence maps for each frame, initialized as empty
        self.processed_images = np.empty((0, 0, 0, 3))  # Array to store processed images for each frame (assuming RGB), initialized as empty
        self.intrinsics = np.empty((0, 3, 3))  # Array to store camera intrinsics for each frame, initialized as empty

        self.logger = logging.getLogger(__name__)

    # -----------------------------
    # Special Methods
    # -----------------------------    
    
    def __repr__(self) -> str:
        """
        Get a string representation of the PointcloudOutputs instance, including the number of frames currently stored in the outputs.

        :param self: The instance of the PointcloudOutputs class.
        :type self
        :return: A string representation of the PointcloudOutputs instance.
        :rtype: str
        """

        return f"PointcloudOutputs(num_frames={len(self.depth)})"
    
    def __str__(self) -> str:
        """
        Get a human-readable string representation of the PointcloudOutputs instance, including the number of frames currently stored in the outputs.

        :param self: The instance of the PointcloudOutputs class.
        :type self
        :return: A human-readable string representation of the PointcloudOutputs instance.
        :rtype: str
        """

        return f"PointcloudOutputs with {len(self.depth)} frames of depth, confidence, and processed images"
    
    def __bool__(self) -> bool:
        """
        Get a boolean value indicating whether there are any frames currently stored in the outputs.

        :param self: The instance of the PointcloudOutputs class.
        :type self
        :return: True if there is at least one frame currently stored in the outputs, False otherwise.
        :rtype: bool
        """

        return len(self.depth) > 0
    
    def __call__(self, window_depth: np.ndarray, window_conf: np.ndarray, window_processed_images: np.ndarray, window_intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update the outputs with new depth maps, confidence maps, and processed images from a new prediction for a window of frames.

        :param self: The instance of the PointcloudOutputs class.
        :type self
        :param window_depth: The new depth maps for the window of frames to be integrated into the overall outputs.
        :type window_depth: np.ndarray
        :param window_conf: The new confidence maps for the window of frames to be integrated into the overall outputs.
        :type window_conf: np.ndarray
        :param window_processed_images: The new processed images for the window of frames to be integrated into the overall outputs.
        :type window_processed_images: np.ndarray
        :param window_intrinsics: The new camera intrinsics for the window of frames to be integrated into the overall outputs.
        :type window_intrinsics: np.ndarray
        """

        # 0) If the outputs are currently empty, simply initialize them with the new window of frames
        if len(self) == 0:
            self._append(window_depth, window_conf, window_processed_images, window_intrinsics)

        else:
            # 1) Get the start index of the overlapping region in the overall prediction,
            # which is determined by the length of the new window of frames (negative integer index),
            # or the length of the current outputs, whichever is smaller
            overlap_start_frame_idx = -min(len(window_depth) - 1, len(self))

            # 2) Get a boolean mask indicating where the confidence in the new prediction is higher than the confidence in the last prediction
            replacement_mask = self._get_overlap_update_mask(overlap_start_frame_idx, window_conf[:-1])

            # 2) Update the overall depth and confidence maps and processed images in the overlapping region based on the new confidence values
            self._update_pointcloud_data_on_overlap(overlap_start_frame_idx, replacement_mask, window_depth[:-1], window_conf[:-1], window_processed_images[:-1])
            
            # 3) Append the new frame's depth, confidence, and processed image to the overall outputs
            self._append(window_depth[-1:], window_conf[-1:], window_processed_images[-1:], window_intrinsics[-1:])

        # 4) Return the updated overall depth, confidence, and processed images after integrating the new prediction
        return self.depth, self.conf, self.processed_images

    def __len__(self) -> int:
        """
        Get the number of frames currently stored in the outputs.

        :param self: The instance of the PointcloudOutputs class.
        :type self
        :return: The number of frames currently stored in the outputs.
        :rtype: int
        """

        return len(self.depth)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the outputs for a specific frame by index.

        :param self: The instance of the PointcloudOutputs class.
        :type self
        :param idx: The frame index to get the outputs for.
        :type idx: int
        :return: A tuple containing the depth map, confidence map, and processed image for the specified frame.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        if not (0 <= idx < len(self)):
            raise IndexError(f"Frame index {idx} is out of range for outputs with {len(self)} frames")
        return self.depth[idx], self.conf[idx], self.processed_images[idx]
    
    # -----------------------------
    # Overall pointcloud data methods
    # -----------------------------

    @property
    def dynamic_conf_thresh(self) -> float:
        """
        Compute the confidence threshold for filtering points in the point cloud based on the confidence maps of the predictions and the configured percentiles.

        :param self: The instance of the PointcloudOutputs class.
        :type self
        :return: The computed confidence threshold for filtering points in the point cloud.
        :rtype: float
        """

        if not self:
            raise ValueError("Cannot compute confidence threshold for empty outputs")
        
        lower = np.percentile(self.conf, self.conf_thresh_percentile)
        upper = np.percentile(self.conf, self.ensure_thresh_percentile)
        conf_thresh = min(max(self.conf_thresh, lower), upper)
        return conf_thresh
    
    # -----------------------------
    # __call__ helpers
    # -----------------------------

    def _append(self, depth: np.ndarray, conf: np.ndarray, processed_images: np.ndarray, intrinsics: np.ndarray) -> None:

        """
        Append new depth maps, confidence maps, and processed images to the overall outputs.

        :param self: The instance of the PointcloudOutputs class.
        :type self
        :param depth: The new depth maps to be appended to the overall outputs.
        :type depth: np.ndarray
        :param conf: The new confidence maps to be appended to the overall outputs.
        :type conf: np.ndarray
        :param processed_images: The new processed images to be appended to the overall outputs.
        :type processed_images: np.ndarray
        :param intrinsics: The new camera intrinsics to be appended to the overall outputs.
        :type intrinsics: np.ndarray

        """
        if len(self) == 0:
            self.depth = np.empty((0, *depth.shape[1:]))
            self.conf = np.empty((0, *conf.shape[1:]))
            self.processed_images = np.empty((0, *processed_images.shape[1:]))
            self.intrinsics = np.empty((0, *intrinsics.shape[1:]))

        self.depth = np.concatenate((self.depth, depth), axis=0)
        self.conf = np.concatenate((self.conf, conf), axis=0)
        self.processed_images = np.concatenate((self.processed_images, processed_images), axis=0)
        self.intrinsics = np.concatenate((self.intrinsics, intrinsics), axis=0)

    def _get_overlap_update_mask(self, overlap_start_frame_idx: int, new_conf: np.ndarray) -> np.ndarray:

        """
        Get a boolean mask for the overlapping region of the last and new predictions, indicating where the confidence in the new prediction is higher than the confidence in the last prediction.

        :param self: The instance of the PointcloudOutputs class.
        :type self
        :param overlap_start_frame_idx: The index of the first frame in the overlapping region in the overall prediction (negative integer).
        :type overlap_start_frame_idx: int
        :param new_conf: The new confidence maps for the overlapping region and the new frame in the window.
        :type new_conf: np.ndarray
        :return: A boolean numpy array of the same shape as the overlapping region, where True indicates that the confidence in the new prediction is higher than the confidence in the last prediction.
        :rtype: np.ndarray
        """

        if not self:
            raise ValueError("Cannot get new better mask for overlap when outputs are empty")
        
        last_overlap_conf = self.conf[overlap_start_frame_idx:]
        new_overlap_conf = new_conf
        
        mask = new_overlap_conf > last_overlap_conf
        
        return mask
    
    def _update_pointcloud_data_on_overlap(self, overlap_start_frame_idx: int, overlap_update_mask: np.ndarray, overlap_new_depth: np.ndarray, overlap_new_conf: np.ndarray, overlap_new_processed_images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Update the overall depth and confidence maps and processed images in the overlapping region based on the new confidence values.

        :param self: The instance of the PointcloudOutputs class.
        :type self
        :param overlap_start_frame_idx: The start index of the overlapping region in the overall prediction as a negative integer.
        :type overlap_start_frame_idx: int
        :param overlap_update_mask: A boolean numpy array of the same shape as the overlapping region, where True indicates that the confidence in the new prediction is higher than the confidence in the last prediction.
        :type overlap_update_mask: np.ndarray
        :param overlap_new_depth: The new depth maps for the overlapping region.
        :type overlap_new_depth: np.ndarray
        :param overlap_new_conf: The new confidence maps for the overlapping region.
        :type overlap_new_conf: np.ndarray
        :param overlap_new_processed_images: The new processed images for the overlapping region.
        :type overlap_new_processed_images: np.ndarray
        :return: A tuple containing the updated overall depth maps, confidence maps, and processed images after integrating the new prediction in the overlapping region.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        self.depth[overlap_start_frame_idx:][overlap_update_mask] = overlap_new_depth[overlap_update_mask]
        self.conf[overlap_start_frame_idx:][overlap_update_mask] = overlap_new_conf[overlap_update_mask]
        self.processed_images[overlap_start_frame_idx:][overlap_update_mask] = overlap_new_processed_images[overlap_update_mask]

        return self.depth, self.conf, self.processed_images