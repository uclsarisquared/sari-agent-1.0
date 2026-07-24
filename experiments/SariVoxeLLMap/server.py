"""
Still being fixed

Currently the topmost module
"""

import socketio, os, asyncio, logging
from aiohttp import web
from time import time
import numpy as np
from pointcloud import *
from voxelization import *
from save import *
from locator import *

class Server:

    def __init__(self):
        """
        Initializes the server and sets up necessary data structures for handling incoming frame data and generating world points.
        """

        # Core object for producing/updating pointcloud
        self.dynamic_pointcloud = None
        self.voxelizator = None

        self.save_dir = ""

        # Main queue task for frame queuing and processing
        self.frame_queue_task = None
        self.save_queue_task = None # For generating and saving plots (use multiple workers)

        # Flags for saving intermediate results for debugging/visualization purposes; can be set to False to save time and storage during normal operation
        self.save_glb = False
        self.save_preprocessed = False
        self.save_no_outlier = False
        self.save_no_floor_ceil = True
        self.save_histogram = True
        self.save_heightmap = True

        # Socket.IO server setup
        self.sio = socketio.AsyncServer(async_mode='aiohttp', logger=False, engineio_logger=False)
        self.app = web.Application()
        self.sio.attach(self.app)

        # Event handlers for client connections, disconnections, and data reception
        self.sio.on('connect', self.on_client_connect)
        self.sio.on('disconnect', self.on_client_disconnect)
        self.sio.on('ping_from_client', self.on_ping_from_client)
        self.sio.on('run_id', self.run_id)
        self.sio.on('frame_data', self.frame_data)

        # Configure root logger once for the whole app.
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)-5s] %(message)s',
            force=True,
        )
        self.logger = logging.getLogger(__name__)

    async def index(self, request):
    	# TODO: check
        with open('latency.html') as f:
            return web.Response(text=f.read(), content_type='text/html')

    async def on_client_connect(self, sid, environ):
        """
        On client connect, request initial run and frame data.

        :param sid: The session ID of the connected client.
        :param environ: The environment information of the connection.
        """
        self.logger.info(f"Client {sid} connected.")

    async def run_id(self, sid, run_id):
        """
        Handle the 'run_id' event sent by the client, which contains information about the current run or session.
        Used to initialize or reset the dynamic point cloud and the background frame queueing task for a new run.
        
        :param sid: The session ID of the client that sent the run_id event.
        :param run_id: The unique identifier for the run or session sent by the client.
        """
        self.dynamic_pointcloud = DynamicPointcloud(save_dir=os.path.join('runs', run_id))
        self.voxelizator = Voxelizator()
        self.logger.info(f"Received run_id: {run_id}")
        
        # For parallel reception and processing of frames
        if self.frame_queue_task is None or self.frame_queue_task.done():

            # Start the background task to process the frame queue
            self.frame_queue_task = asyncio.create_task(self._process_frame_queue())
            self.logger.info(f"Frame processing task started for run_id: {run_id}")

            # Start the background task to process the saving of files (e.g., plots, 3d models, arrays) if needed
            self.save_queue_task = SaveTask(4)
            self.save_queue_task.clear_staged()
            self.logger.info(f"Save processing task started for run_id: {run_id}")

    async def frame_data(self, sid, data):
        """
        Handle the 'frame_data' event sent by the client by queuing incoming frame data for asynchronous processing into world points using the dynamic point cloud model.
        
        :param sid: The session ID of the client that requested the world points data.
        :param data: The data sent by the client, which may include parameters for generating the world points (e.g., image directory, camera parameters).
        """

        assert self.dynamic_pointcloud is not None and self.voxelizator is not None, "Run data must be sent before frame data."
        
        # Remove comment for simple synchronous processing
        #self.dynamic_pointcloud(data['image_path'], np.array(data['extrinsics']), np.array(data['intrinsics']))        
         
        # For parallel reception and processing of frames
        self.dynamic_pointcloud.frame_queue.put_nowait((data['image_path'], np.array(data['extrinsics']), np.array(data['intrinsics'])))
        frame_name = data['image_path'].split('/')[-1]
        self.logger.info(f"Frame received: {frame_name}. Queue size: {self.dynamic_pointcloud.frame_queue.qsize()}")
	
    async def _process_frame_queue(self):
        """
        Background task that continually consumes frame tuples from the queue
        and spawns pointcloud updates in worker threads to avoid blocking the event loop.
        """

        while True:
            if self.dynamic_pointcloud and self.voxelizator:

                self.save_dir = self.dynamic_pointcloud.save_dir

                # 1) Wait for the next frame tuple
                image_path, extrinsics, intrinsics = await self.dynamic_pointcloud.frame_queue.get()

                start = time()

                # 1a) Extract frame name for logging and saving purposes
                filename = image_path.split('/')[-1]
                name = filename.split('.')[0]
                self.logger.info(f"Processing frame: {name}. Queue size: {self.dynamic_pointcloud.frame_queue.qsize()}")

                # 2) Process the frame in a separate thread to avoid blocking the event loop
                start_ = time()
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

                    # 5) Emit the updated heightmap back to the client
                    await self.sio.emit('heightmap', {
                        'data': format_back_to_unity_coordinates(
                            self.voxelizator.heightmap,
                            self.voxelizator.xedges,
                            self.voxelizator.yedges
                        ),
                        'min_z': float(self.voxelizator.min_z),
                        'bin_width': float(self.voxelizator.bin_width)
                    })
                    self.logger.info(f"Heightmap emitted to client. Time: {time() - start:.2f} seconds.")
                    self.logger.info("=" * 60)

                # 5) Mark the frame as processed in the queue
                self.dynamic_pointcloud.frame_queue.task_done() 
                

    async def on_client_disconnect(self, sid, reason):
        """
        Handle client disconnection.    

        :param sid: The session ID of the disconnected client.
        :param reason: The reason for the client's disconnection.
        """

        self.logger.info(f"Client {sid} disconnected: {reason}")
        
        assert self.dynamic_pointcloud is not None and self.voxelizator is not None, "Run data must be sent before disconnecting."
        assert self.save_queue_task is not None, "Save queue task must be initialized before processing frames."

        # Finish processing all received frames before flushing deferred saves.
        await self.dynamic_pointcloud.frame_queue.join()

        # Start queued save jobs only at disconnect.
        self.save_queue_task.flush_staged()

        # Additionally, save the final state of the point cloud and voxelization results for this run, which can be useful for debugging or visualization purposes.
        self.save_queue_task(
            save_npz,
            intrinsics=self.dynamic_pointcloud.outputs.intrinsics,
            extrinsics=self.dynamic_pointcloud.inputs.extrinsics,
            depths=self.dynamic_pointcloud.outputs.depth,
            confs=self.dynamic_pointcloud.outputs.conf,
            points=self.dynamic_pointcloud.points,
            colors=self.dynamic_pointcloud.colors,
            preprocessed_points=self.voxelizator.preprocessed_points,
            no_outlier_points=self.voxelizator.no_outlier_points,
            no_floor_ceil_points=self.voxelizator.no_floor_ceil_points,
            histogram2d=self.voxelizator.histogram2d,
            xedges=self.voxelizator.xedges,
            yedges=self.voxelizator.yedges,
            height_estimates=self.voxelizator.height_estimates,
            heightmap=self.voxelizator.heightmap,
            save_dir=self.save_dir,
            suffix="final",
            delete_old=False
        )

        # Cancel the frame processing task — no more frames will arrive.
        if self.frame_queue_task is not None and not self.frame_queue_task.done(): 
            self.frame_queue_task.cancel()
            self.frame_queue_task = None

        # Wait for all queued save jobs to complete before shutting down the executor.
        await self.save_queue_task.queue.join()
        self.save_queue_task.shutdown()
        self.save_queue_task = None
        self.logger.info("All save jobs completed.")

        # Print the final state of the point cloud and voxelization results for debugging purposes.
        self.logger.info(f"Final point cloud state: {len(self.dynamic_pointcloud.points)} points")
        self.logger.info(f"Final histogram shape: {self.voxelizator.histogram2d.shape if self.voxelizator.histogram2d is not None else 'None'}")

        # Exit the server if needed (e.g., if only one client is expected)
        os._exit(0)

    async def on_ping_from_client(self, sid):
        """
        For handling 'ping_from_client' events sent by the client. Upon receiving this event, the server will respond with a 'pong_from_server' event, which can be used by the client to measure latency.
        
        :param sid: The session ID of the client that sent the ping event.
        """
        await self.sio.emit('pong_from_server', room=sid)

# -----------------------
# Main entry point
# -----------------------

if __name__ == '__main__':
    server = Server()
    web.run_app(server.app, host='localhost', port=5050)
