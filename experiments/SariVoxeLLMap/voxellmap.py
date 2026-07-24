"""
SariVoxeLLMap: A class that manages the dynamic point cloud and voxelization to create a 2D heightmap representation of the environment for use in an agent's world model.

It includes methods for
- initializing a run
- queuing frames for processing
- retrieving the current map representation.

The class also handles background processing of frames and saving intermediate results for debugging and visualization purposes.

For usage, 
1) Call `init_run(run_id)` once at the start of a run to set up the point cloud and voxelizator.
2) Call `queue_frame(img_path, extrinsics, intrinsics)` to add frames for
    processing as the agent moves and collects data.
3) Call `get_current_map(current_position)` to retrieve the current map representation, optionally with coordinates relative to the agent's current position.

Upcoming features include:
- Getting current map as top-down heightmap image with agent position marked
- Getting depth estimate for pixel coordinates in the current view, either by:
    - refering / waiting for and referring to results for most recently queued frame
    - or by calculating on the fly from current point cloud and voxelizator results

- Calculating extrinsics for either:
    - a given position and orientation
    - transformation and original extrinsics
"""

import asyncio, os, logging
from time import time
import numpy as np
from pointcloud import *
from voxelization import *
from save import *

class SariVoxeLLMap:

    def __init__(self):

         # Core processing objects
        self.dynamic_pointcloud = None
        self.voxelizator = None
        
        # Background tasks
        self.frame_process_task = None
        self.save_queue_task = None

        # Flags for saving intermediates
        self.save_dir = ""
        self.save_glb = False
        self.save_preprocessed = False
        self.save_no_outlier = False
        self.save_no_floor_ceil = True
        self.save_histogram = True
        self.save_heightmap = True

        # Configure logger
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)-5s] %(message)s',
            force=True,
        )
        self.logger = logging.getLogger(__name__)

    async def init_run(self, run_id: str):
        """
        Initialize a new run: create point cloud, voxelizator, and start background tasks.
        Call this once before requesting frames.
        """
        self.manual_depth_processor = DA3Wrapper()
        self.dynamic_pointcloud = DynamicPointcloud(save_dir=os.path.join('runs', run_id))
        self.voxelizator = Voxelizator()
        self.logger.info(f"Initialized run: {run_id}")

        if self.frame_process_task is None or self.frame_process_task.done():
            self.frame_process_task = asyncio.create_task(self._process_frame_queue())
            self.logger.info("Frame processing task started")

        if self.save_queue_task is None:
            self.save_queue_task = SaveTask(4)
            self.save_queue_task.clear_staged()
            self.logger.info("Save queue task started")

    async def shutdown(self):
        """
        Clean shutdown: stop requesting frames and finish processing.
        """
        if self.dynamic_pointcloud:
            await self.dynamic_pointcloud.frame_queue.join()
        if self.save_queue_task:
            await self.save_queue_task.queue.join()
            self.save_queue_task.shutdown()
        if self.frame_process_task and not self.frame_process_task.done():
            self.frame_process_task.cancel()
        self.logger.info("Server shutdown complete")

    # -------------------------------------------
    # For external calls to queue frames for processing
    # -------------------------------------------

    async def queue_frame(self, img_path: str, extrinsics: np.ndarray, intrinsics: np.ndarray) -> int:
        """
        Appends the frame to the queue for processing for inclusion in the map.
        Can be called right within movement tool calls

        :param img_path: The path of the image whose depth will be predicted
        :type img_path: str
        :param extrinsics: The extrinsics array at the position at which the image is taken (4, 4)
        :type extrinsics: np.ndarray
        :param intrinsics: The intrinsics array of the camera when the image is taken (3, 3)
        :type intrinsics: np.ndarray        
        :return: The size of the frame queue after adding the frame
        :rtype: int
        """

        # Queue for processing
        if self.dynamic_pointcloud:
            self.dynamic_pointcloud.frame_queue.put_nowait((
                img_path,
                extrinsics,
                intrinsics
            ))
            qsize = self.dynamic_pointcloud.frame_queue.qsize()
            self.logger.info(f"Frame queued: {img_path}. Queue size: {qsize}")  
            return qsize
        else:
            self.logger.warning("Dynamic point cloud not initialized. Call init_run() first.")
            return -1                

    # -------------------------------------------
    # For agent usage
    # -------------------------------------------

    async def get_current_map(self, current_position: tuple | None) -> list[tuple[float, float, float]]:
        """
        Get the current map representation, with coordinates optionally relative to the current position.

        :param current_position: The (x, y, z) coordinates of the current position
        :type current_position: tuple
        :return: A list of tuples containing the (x, y, z) coordinates of the points in the current map
        :rtype: list[tuple[float, float, float]]
        """
        if self.voxelizator is None:
            self.logger.warning("Voxelizator not initialized. Call init_run() first.")
            return []

        heightmap = format_as_list_of_coordinates(
                        self.voxelizator.heightmap,
                        self.voxelizator.xedges,
                        self.voxelizator.yedges
                    )
        
        # For when the agent asks for the map with coordinates relative to its current position (e.g., for use in world model)
        if current_position is not None:
            x, y, z = current_position
            rel_heightmap = []
            for vx, vy, vz in heightmap:
                rel_heightmap.append((vx - x, vy - y, vz - z))
            heightmap = rel_heightmap
        
        return heightmap

    # -------------------------------------------
    # Background processing of frames and saving results
    # -------------------------------------------

    async def _process_frame_queue(self):
        """
        Background task that processes queued frames (same as before).
        Consumes from frame queue, updates pointcloud, saves results.
        """
        while True:
            try:
                if self.dynamic_pointcloud is None or self.voxelizator is None:
                    await asyncio.sleep(0.5)
                    continue

                self.save_dir = self.dynamic_pointcloud.save_dir
                image_path, extrinsics, intrinsics = await self.dynamic_pointcloud.frame_queue.get()

                start = time()
                filename = image_path.split('/')[-1]
                name = filename.split('.')[0]
                self.logger.info(f"Processing frame: {name}. Queue size: {self.dynamic_pointcloud.frame_queue.qsize()}")

                # Process frame in thread pool (existing logic)
                data = await asyncio.to_thread(self.dynamic_pointcloud, image_path, extrinsics, intrinsics)

                if data and self.dynamic_pointcloud.points is not None and len(self.dynamic_pointcloud.points) > 0:
                    assert self.save_queue_task is not None, "Save queue task must be initialized before staging saves."
                    
                    # 3) If per-frame tracking of 3D model updates is needed, save the updated point cloud for this frame (e.g., for visualization or debugging purposes)
                    if self.save_glb == True:
                        self.save_queue_task.stage(
                            save_scene_glb,
                            extrinsics = np.copy(data['extrinsics']),
                            intrinsics = np.copy(data['intrinsics']),
                            depth_shape = data['depth_shape'],
                            points = np.copy(data['points']),
                            colors = np.copy(data['colors']),
                            # Path
                            save_dir = self.dynamic_pointcloud.save_dir,
                            suffix = name,
                            delete_old = False
                        )
                    
                    # 4) After processing, convert the updated point cloud to a 2D grid
                    self.voxelizator.preprocessed_points = await asyncio.to_thread(preprocess_points, data['points'])
                    if self.save_preprocessed == True:
                        self.save_queue_task.stage(
                            save_scatter,
                            np.copy(self.voxelizator.preprocessed_points),
                            save_dir = self.save_dir,
                            prefix = name,
                            suffix = "1_preprocessed_points",
                            title = f"Preprocessed points (Frame {name})",
                            delete_old = False
                        )
                    
                    self.voxelizator.no_outlier_points = await asyncio.to_thread(remove_outliers, self.voxelizator.preprocessed_points)
                    if self.save_no_outlier == True:
                        self.save_queue_task.stage(
                            save_scatter,
                            np.copy(self.voxelizator.no_outlier_points),
                            save_dir = self.save_dir,
                            prefix = name,
                            suffix = "2_no_outlier_points",
                        title = f"No-outlier points (Frame {name})",
                        delete_old = False
                    )
                    
                    self.voxelizator.no_floor_ceil_points = await asyncio.to_thread(remove_floor_ceil, self.voxelizator.no_outlier_points)
                    if self.save_no_floor_ceil == True:
                        self.save_queue_task.stage(
                        save_scatter,
                        np.copy(self.voxelizator.no_floor_ceil_points),
                        save_dir = self.save_dir,
                        prefix = name,
                        suffix = "3_no_floor_ceil_points",
                        title = f"No-floor-and-ceiling points (Frame {name})",
                        delete_old = False
                    )

                    x, y, z = split_axes(self.voxelizator.no_floor_ceil_points) 
                    self.voxelizator.histogram2d, self.voxelizator.xedges, self.voxelizator.yedges, self.voxelizator.bin_width = await asyncio.to_thread(get_histogram2d, x, y, z)

                    # Compute height estimates and density mask concurrently; both depend on histogram, not on each other.
                    height_task = asyncio.to_thread(
                        get_height_estimates,
                        np.copy(x),
                        np.copy(y),
                        np.copy(z),
                        np.copy(self.voxelizator.histogram2d),
                        np.copy(self.voxelizator.xedges),
                        np.copy(self.voxelizator.yedges),
                    )
                    
                    density_mask_task = asyncio.to_thread(
                        get_density_filter_mask,
                        np.copy(self.voxelizator.histogram2d),
                    )

                    (
                        (self.voxelizator.height_estimates, self.voxelizator.min_z),
                        density_mask,
                    ) = await asyncio.gather(height_task, density_mask_task)

                    height_mask = await asyncio.to_thread(
                        get_height_filter_mask,
                        np.copy(self.voxelizator.height_estimates),
                    )

                    if density_mask is None:
                        density_mask = np.ones_like(self.voxelizator.histogram2d, dtype=bool)
                    if height_mask is None:
                        height_mask = np.ones_like(self.voxelizator.height_estimates, dtype=bool)

                    if self.save_histogram == True:
                        self.save_queue_task.stage(
                            #save_bar3d,
                            save_2d_grid,
                            np.copy(self.voxelizator.height_estimates),
                            #np.copy(self.voxelizator.histogram2d),
                            np.copy(self.voxelizator.xedges),
                            np.copy(self.voxelizator.yedges),
                            #np.copy(self.voxelizator.min_z),
                            #np.copy(self.voxelizator.bin_width),
                            save_dir = self.save_dir,
                            prefix = name,
                            suffix = "4_heights",
                            title = f"Heights (Frame {name})",
                            delete_old = False
                        )

                    self.voxelizator.heightmap = await asyncio.to_thread(
                        get_filtered_heightmap,
                        np.copy(self.voxelizator.height_estimates),
                        np.copy(density_mask),
                        np.copy(height_mask)
                    )
                    
                    if self.save_heightmap == True:
                        self.save_queue_task.stage(
                            #save_bar3d,
                            save_2d_grid,
                            np.copy(self.voxelizator.height_estimates),
                            #np.copy(self.voxelizator.histogram2d),
                            np.copy(self.voxelizator.xedges),
                            np.copy(self.voxelizator.yedges),
                            #np.copy(self.voxelizator.min_z),
                            #np.copy(self.voxelizator.bin_width),
                            save_dir = self.save_dir,
                            prefix = name,
                            suffix = "5_filtered_heights",
                            title = f"Filtered Heights (Frame {name})",
                            delete_old = False
                        )

                    self.logger.info(f"Frame processed in {time() - start:.2f}s")

                self.dynamic_pointcloud.frame_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
