import os, logging, glob
import numpy as np
from PIL import Image
from threading import Thread
# ----------------------------------------------

class PointcloudInputs:

    def __init__(self, save_dir: str, num_frame_overlap: int = 3) -> None:
        """
        Initialize the PointcloudInputs class to manage the inputs for dynamic point cloud prediction.

        :param num_frame_overlap: The number of overlapping frames to include in each window of frames for inference.
        :type num_frame_overlap: int
        :param save_dir: The directory where inputs should be saved.
        :type save_dir: str
        """    
    
        self.num_frame_overlap = num_frame_overlap
        self.save_dir = save_dir

        # Create the save directory if it does not exist
        os.makedirs(self.save_dir, exist_ok=True)

        self.image_paths = []  # List to store file paths of input images, ordered by timestamp or filename
        self.image_arrays = np.empty((0, 0, 0, 3), dtype=np.uint8)  # Array to store the actual image data for each frame (N, H, W, C), initialized as empty
        self.extrinsics = np.empty((0, 4, 4))  # Array to store extrinsic camera parameters (world-to-camera) for each frame, initialized as empty
        self.intrinsics = np.empty((0, 3, 3))  # Array to store intrinsic camera parameters for each frame, initialized as empty

        self.logger = logging.getLogger(__name__)

    # -----------------------------
    # Special Methods
    # -----------------------------    
    
    def __repr__(self) -> str:
        """
        Get a string representation of the PointcloudInputs instance, including the number of frames currently stored in the inputs.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: A string representation of the PointcloudInputs instance.
        :rtype: str
        """

        return f"PointcloudInputs(num_frames={len(self.image_paths)}, num_frame_overlap={self.num_frame_overlap})"
    
    def __bool__(self) -> bool:
        """
        Get a boolean value indicating whether there are any frames currently stored in the inputs.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: True if there is at least one frame currently stored in the inputs, False otherwise.
        :rtype: bool
        """

        return len(self.image_paths) > 0

    def __call__(self, image_path: str, extrinsics: np.ndarray, intrinsics: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Add new inputs for a single frame and get the current window of frames for inference.
        This method should be called after each new frame is added to the input directory and before running inference on the latest window of frames,
        to ensure that the latest inputs are included in the prediction.
        
        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: The current window of frames for inference, including the list of image file paths, the extrinsics array, and the intrinsics array.
        :rtype: tuple[list[str], np.ndarray, np.ndarray]
        """

        # 1) Append the new frame's inputs to the current set of inputs
        self.append(image_path, extrinsics, intrinsics)

        # 2) Get the current window of frames for inference, which includes the latest frame and the specified number of overlapping frames before it        
        return self.active_window
    
    def __len__(self) -> int:
        """
        Get the number of frames currently stored in the inputs.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: The number of frames currently stored in the inputs.
        :rtype: int
        """

        return len(self.image_paths)
    
    def __getitem__(self, idx: str | int) -> tuple[str, np.ndarray, np.ndarray]:
        """
        Get the inputs for a specific frame by index or image path.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :param idx: The index of the frame or the image path for which to get the inputs.
        :type idx: str | int
        :return: A tuple containing the image file path, extrinsics array, and intrinsics array for the specified frame index.
        :rtype: tuple[str, np.ndarray, np.ndarray]
        """

        if isinstance(idx, str):
            if idx not in self.image_paths:
                raise KeyError(f"Image path '{idx}' not found in inputs")
            idx = self.image_paths.index(idx)  # Convert image path to index if a string is provided
        else:
            if not (0 <= idx < len(self.image_paths)):
                raise IndexError(f"Frame index {idx} out of range for inputs with {len(self.image_paths)} frames")
        return self.image_paths[idx], self.extrinsics[idx], self.intrinsics[idx]
    
    def __contains__(self, id: str | int) -> bool:
        """
        Check if a specific image path or frame index is currently stored in the inputs.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :param id: The image path or frame index to check for presence in the inputs.
        :type id: str | int
        :return: True if the specified image path or frame index is currently stored in the inputs, False otherwise.
        :rtype: bool
        """

        if isinstance(id, str):
            return id in self.image_paths
        else:
            return 0 <= id < len(self.image_paths)
        
    # ---------------------------------------------
    # Get specific subsets of inputs for inference
    # ---------------------------------------------

    @property
    def last(self) -> tuple[str, np.ndarray, np.ndarray]:
        """
        Get the inputs for the latest frame currently stored in the inputs.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: A tuple containing the image file path, extrinsics array, and intrinsics array for the latest frame currently stored in the inputs.
        :rtype: tuple[str, np.ndarray, np.ndarray]
        """

        return self[-1]  # Get the last frame's data using negative indexing

    @property
    def active_window(self) -> tuple[list[str], np.ndarray, np.ndarray]:

        """
        Get the list of image file paths and corresponding extrinsics/intrinsics for the current window of frames to be used for inference.
        This should be called before running inference on the latest window of frames, to ensure that the correct inputs are provided to the model.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: A tuple containing the list of image file paths, the extrinsics array, and the intrinsics array for the current window of frames.
        :rtype: tuple[list[str], np.ndarray, np.ndarray]
        """    
        
        num_frames_to_process = min(self.num_frame_overlap + 1, len(self))  # +1 because the latest frame is included in the window  

        return self.image_paths[-num_frames_to_process:], self.extrinsics[-num_frames_to_process:], self.intrinsics[-num_frames_to_process:]
    
    @property
    def window_image_paths(self) -> list[str]:
        """
        Get the list of image file paths for the current window of frames to be used for inference.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: The list of image file paths for the current window of frames.
        :rtype: list[str]
        """

        return self.active_window[0]
    
    @property
    def window_extrinsics(self) -> np.ndarray:
        """
        Get the extrinsics array for the current window of frames to be used for inference.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: The extrinsics array for the current window of frames.
        :rtype: np.ndarray
        """

        return self.active_window[1]
    
    @property
    def window_intrinsics(self) -> np.ndarray:
        """
        Get the intrinsics array for the current window of frames to be used for inference.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: The intrinsics array for the current window of frames.
        :rtype: np.ndarray
        """

        return self.active_window[2]

    @property
    def all(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Get all inputs currently stored in the PointcloudInputs instance.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: A tuple containing the list of image file paths, the extrinsics array, and the intrinsics array for all frames currently stored in the inputs.
        :rtype: tuple[list[str], np.ndarray, np.ndarray]
        """

        return self.image_paths, self.extrinsics, self.intrinsics
    
    # ---------------------------------------------
    # Methods for modifying the inputs
    # ---------------------------------------------
    
    def append(self, image_path: str, extrinsics: np.ndarray, intrinsics: np.ndarray) -> None:
        """
        Add new inputs for a single frame to the current set of inputs.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :param image_path: The file path of the image for the new frame to be added.
        :type image_path: str
        :param extrinsics: The extrinsics array for the new frame to be added, with shape (4, 4).
        :type extrinsics: np.ndarray
        :param intrinsics: The intrinsics array for the new frame to be added, with shape (3, 3).
        :type intrinsics: np.ndarray
        :return: None
        :rtype: None
        """    

        # 1) Append the new frame's image path
        self.image_paths.append(image_path)

        # 2) Convert the image into an array and append it to the image_arrays attribute if image_type is set to "arrays"
        #while not os.path.exists(image_path):
        #    pass
        #pil_image = Image.open(image_path)
        #pil_image.load()
        #np_image = np.asarray(pil_image, dtype=np.uint8)
        #if self.image_arrays.shape[0] == 0:
        #    self.image_arrays = np_image[np.newaxis]  # Initialize the image_arrays with the first image
        #else:
        #    self.image_arrays = np.concatenate((self.image_arrays, np_image[np.newaxis]), axis=0)

        # 3) Append the new frame's extrinsics
        assert extrinsics.shape == (4, 4), "Extrinsics for each frame must have shape (4, 4)"
        self.extrinsics = np.concatenate((self.extrinsics, extrinsics[np.newaxis]), axis=0)

        # 4) Append the new frame's intrinsics
        assert intrinsics.shape == (3, 3), "Intrinsics must be provided for each frame and have shape (3, 3)"
        self.intrinsics = np.concatenate((self.intrinsics, intrinsics[np.newaxis]), axis=0)


    def clear(self) -> None:
        """
        Clear all inputs currently stored in the PointcloudInputs instance.

        :param self: The instance of the PointcloudInputs class.
        :type self
        :return: None
        :rtype: None
        """

        self.image_paths = []
        self.extrinsics = np.empty((0, 4, 4))
        self.intrinsics = np.empty((0, 3, 3))
        self.image_arrays = np.empty((0, 3, 3), dtype=np.uint8)

        self.logger.info("Cleared all inputs from PointcloudInputs instance")