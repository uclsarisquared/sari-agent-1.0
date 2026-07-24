import sys, os, glob, trimesh, traceback, logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("C:\\Users\\Kat Gados\\Documents\\Thesis\\Code\\2526\\agent\\Depth-Anything-3\\src\\depth_anything_3")  # Adjust the path as needed to access DA3 modules
from depth_anything_3.utils.export.glb import _filter_and_downsample, _compute_alignment_transform_first_cam_glTF_center_by_points, _estimate_scene_scale, _add_cameras_to_scene

def prep_dir_and_get_filepath(
    save_dir: str,
    prefix: str,
    suffix: str,
    ext: str,
    delete_old: bool   
) -> tuple[str, str]:
    
    """
    Prepares the save directory and gets the filepath

    :param save_dir: Path to the directory where the file will be saved
    :type save_dir: str
    :param prefix: The start of the name of the file to be saved
    :type prefix: str
    :param suffix: The end of the name of the file to be saved
    :type suffix: str
    :param ext: The file extension of the file to be saved
    :type ext: str
    :param delete_old: Whether other files with the same file extention within the directory should be deleted
    :type delete_old: bool
    :return: The filepath without the dot and the file extension and the full filepath
    :rtype: tuple[str, str]
    """
    
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{prefix}_{suffix}"
    filepath_wo_dot_ext, dot_ext = os.path.join(save_dir, filename), f".{ext}"
    filepath = filepath_wo_dot_ext + dot_ext

    if delete_old:
        old_filepaths = glob.glob(os.path.join(save_dir, f"*{dot_ext}"))
        assert filepath not in old_filepaths, f"Expected new file path {filepath} to not already exist in {save_dir}"
        for f in old_filepaths:
            os.remove(f)

    return filepath_wo_dot_ext, filepath

# -----------------------------
# Save Functions
# -----------------------------    

def save_npz(
    # Data
    input_intrinsics: np.ndarray,
    output_intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depths: np.ndarray,
    confs: np.ndarray,
    processed_images: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    preprocessed_points: np.ndarray,
    no_outlier_points: np.ndarray,
    no_floor_ceil_points: np.ndarray,
    histogram2d: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    height_estimates: np.ndarray,
    heightmap: np.ndarray,
    # Save
    save_dir: str,
    suffix: str,
    delete_old: bool,
) -> str:
    
    """
    Saves all of the relevant data for the frame processing to a single compressed numpy file

    :param input_intrinsics: The array of input intrinsics received per frame
    :type input_intrinsics: np.ndarray
    :param output_intrinsics: The array of output intrinsics predicted/used by DA3 per frame
    :type output_intrinsics: np.ndarray
    :param extrinsics: The array of the extrinsics arrays of all frames processed
    :type extrinsics: np.ndarray
    :param depths: The array of the depth arrays of all frames processed
    :type depths: np.ndarray
    :param confs: The array of the confidence arrays of all frames processed
    :type confs: np.ndarray
    :param processed_images: The array of output processed images for all frames
    :type processed_images: np.ndarray
    :param points: The array of the points comprising the pointcloud for the frame processed
    :type points: np.ndarray
    :param colors: The array of the point colors comprising the pointcloud for the frame processed
    :type colors: np.ndarray
    :param preprocessed_points: The array of the preprocessed points comprising the pointcloud for the frame processed
    :type preprocessed_points: np.ndarray
    :param no_outlier_points: The array of the points without outliers comprising the pointcloud for the frame processed
    :type no_outlier_points: np.ndarray
    :param no_floor_ceil_points: The array of the points without the floor and ceiling points comprising the pointcloud for the frame processed
    :type no_floor_ceil_points: np.ndarray
    :param histogram2d: The 2D histogram of point counts in the x-y plane
    :type histogram2d: np.ndarray
    :param xedges: The edges of the bins in the x direction
    :type xedges: np.ndarray
    :param yedges: The edges of the bins in the y direction
    :type yedges: np.ndarray
    :param height_estimates: The unfiltered estimated heights per histogram bin
    :type height_estimates: np.ndarray
    :param heightmap: The array of the (x, y, height) coordinates comprising the voxels for the frame processed
    :type heightmap: np.ndarray
    :param save_dir: Path to the directory where the file will be saved
    :type save_dir: str
    :param suffix: The end of the name of the file to be saved
    :type suffix: str
    :param delete_old: Whether other files with the same file extention within the directory should be deleted
    :type delete_old: bool
    :return: The path to the NPZ file saved
    :rtype: str 
    """
    
    filepath_wo_dot_ext, filepath = prep_dir_and_get_filepath(save_dir, "data", suffix, "npz", delete_old)
    np.savez_compressed(
        filepath_wo_dot_ext,
        # Keep legacy key for backward compatibility.
        intrinsics=output_intrinsics,
        input_intrinsics=input_intrinsics,
        output_intrinsics=output_intrinsics,
        extrinsics=extrinsics,
        depths=depths,
        confs=confs,
        processed_images=processed_images,
        points=points,
        colors=colors,
        preprocessed_points=preprocessed_points,
        no_outlier_points=no_outlier_points,
        no_floor_ceil_points=no_floor_ceil_points,
        histogram2d=histogram2d,
        xedges=xedges,
        yedges=yedges,
        height_estimates=height_estimates,
        heightmap=heightmap
    )
    return filepath
    
def save_scene_glb(
        # Inputs
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        depth_shape: tuple,
        points: np.ndarray,
        colors: np.ndarray,
        # Path
        save_dir: str,
        suffix: str,
        delete_old: bool = False,
        # Size
        num_max_points: int = 1_000_000
) -> str:
    
    """
    Saves a GLB file of a pointcloud

    :param intrinsics: The intrinsics array to be used
    :type intrinsics: np.ndarray
    :param extrinsics: The intrinsics array to be used
    :type extrinsics: np.ndarray
    :param depths: The shape of the depths array corresponding to the point cloud
    :type depths: np.ndarray
    :param points: The array of the points comprising the pointcloud for the frame processed
    :type points: np.ndarray
    :param colors: The array of the point colors comprising the pointcloud for the frame processed
    :type colors: np.ndarray
    :param save_dir: Path to the directory where the file will be saved
    :type save_dir: str
    :param suffix: The end of the name of the file to be saved
    :type suffix: str
    :param delete_old: Whether other files with the same file extention within the directory should be deleted
    :type delete_old: bool
    :param num_max_points: The maximum number of points to be rendered
    :type num_max_points: int
    :return: The path to the GLB file saved
    :rtype: str 
    """
    
    # 1) Based on first camera orientation + glTF axis system, center by point cloud,
    # construct alignment transform, and apply to point cloud
    A = _compute_alignment_transform_first_cam_glTF_center_by_points(
        extrinsics[0], points
    )  # (4,4)

    if points.shape[0] > 0:
        points = trimesh.transform_points(points, A)

    # 2) Clean + downsample
    points, colors = _filter_and_downsample(points, colors, num_max_points)

    # 3) Assemble scene (add point cloud first)
    scene = trimesh.Scene()
    if scene.metadata is None:
        scene.metadata = {}
    scene.metadata["hf_alignment"] = A  # For camera wireframes and external reuse

    if points.shape[0] > 0:
        pc = trimesh.points.PointCloud(vertices=points, colors=colors)
        scene.add_geometry(pc)

    # 4) Draw cameras (wireframe pyramids), using the same transform A
    scene_scale = _estimate_scene_scale(points, fallback=1.0)
    assert len(depth_shape) == 3, "Expected depth shape to be (N, H, W)"
    H, W = depth_shape[1:]
    _add_cameras_to_scene(
        scene=scene,
        K=intrinsics,
        ext_w2c=extrinsics,
        image_sizes=[(H, W)] * depth_shape[0],
        scale=scene_scale * 0.03,
    )

    # 5) Export
    _, filepath = prep_dir_and_get_filepath(save_dir, "scene", suffix, "glb", delete_old)
    scene.export(filepath)
    return filepath

def save_scatter(
        # Inputs
        points: np.ndarray,
        # Path
        save_dir: str,
        prefix: str,
        suffix: str,
        title: str,
        delete_old: bool = False
) -> str | None:
    
    """
    Saves a scatterplot to a PNG file

    :param points: The array of the points comprising the pointcloud for the frame processed
    :type points: np.ndarray
    :param save_dir: Path to the directory where the file will be saved
    :type save_dir: str
    :param prefix: The start of the name of the file to be saved
    :type prefix: str
    :param suffix: The end of the name of the file to be saved
    :type suffix: str
    :param delete_old: Whether other files with the same file extension within the directory should be deleted
    :type delete_old: bool
    :param num_max_points: The maximum number of points to be rendered
    :type num_max_points: int
    :return: The path to the PNG file saved or None if the saving failed
    :rtype: str | None
    """
    
    try:
        matplotlib.use('Agg')  # Use non-interactive backend in thread
        
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(x_coords, y_coords, c=z_coords, cmap='viridis_r')
        fig.colorbar(scatter, ax=ax, label='Max Z')

        ax.set_title(f'{title.title()}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(np.min(x_coords), np.max(x_coords))
        ax.set_ylim(np.min(y_coords), np.max(y_coords))
        ax.set_aspect('equal', adjustable='box')

        _, filepath = prep_dir_and_get_filepath(save_dir, prefix, suffix, "png", delete_old)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')

        plt.close(fig)

        print(f"Saved {filepath}")

        return filepath

    except Exception as e:
        traceback.print_exc()

        try:
            plt.close('all')
        except:
            pass

def save_2d_grid(
        # Inputs
        grid: np.ndarray,
        xticks: np.ndarray,
        yticks: np.ndarray,
        # Path
        save_dir: str,
        prefix: str,
        suffix: str,
        title: str,
        delete_old: bool = False
) -> str | None:
    
    """
    Saves a 2D grids colormap to a PNG file

    :param grid: The array of the 2D grid comprising the heightmap for the frame processed
    :type grid: np.ndarray
    :param xticks: The array of the x-tick values corresponding to the grid
    :type xticks: np.ndarray
    :param yticks: The array of the y-tick values corresponding to the grid
    :type yticks: np.ndarray
    :param save_dir: Path to the directory where the file will be saved
    :type save_dir: str
    :param prefix: The start of the name of the file to be saved
    :type prefix: str
    :param suffix: The end of the name of the file to be saved
    :type suffix: str
    :param delete_old: Whether other files with the same file extention within the directory should be deleted
    :type delete_old: bool
    :param num_max_points: The maximum number of points to be rendered
    :type num_max_points: int
    :return: The path to the PNG file saved or None if the saving failed
    :rtype: str | None
    """
    
    try:
        matplotlib.use('Agg')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid, cmap='viridis_r', interpolation='nearest')
        fig.colorbar(im, ax=ax, label='Height')
        ax.set_title(f'({title.title()})')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_xlim(0, grid.shape[1])
        ax.set_ylim(0, grid.shape[0])
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_aspect('equal', adjustable='box')

        _, filepath = prep_dir_and_get_filepath(save_dir, prefix, suffix, "png", delete_old)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')

        plt.close(fig)

        print(f"Saved {filepath}")

        return filepath

    except Exception as e:
        traceback.print_exc()

        try:
            plt.close('all')
        except:
            pass

def save_bar3d(
        # Inputs
        heightmap: np.ndarray,
        density: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        min_z: float,
        bin_width: float,
        # Path
        save_dir: str,
        prefix: str,
        suffix: str,
        title: str,
        delete_old: bool = False
) -> str | None:
    
    """
    Saves a 3D bar plot to a PNG file

    :param heighmap: The array of the 2D grid comprising of tuples of (x, y, z) the heightmap for the frame processed
    :type heightmap: np.ndarray
    :param density: The array of the density values comprising of tuples of (x, y, density)
    :type density: np.ndarray
    :param xedges: The edges of the bins in the x direction
    :type xedges: np.ndarray
    :param yedges: The edges of the bins in the y direction
    :type yedges: np.ndarray
    :param min_z: The minimum Z-value for the heightmap for the frame processed
    :type min_z: float
    :param bin_width: The width of each bar in the x and y directions for the 3D bar plot
    :type bin_width: float
    :param save_dir: Path to the directory where the file will be saved
    :type save_dir: str
    :param prefix: The start of the name of the file to be saved
    :type prefix: str
    :param suffix: The end of the name of the file to be saved
    :type suffix: str
    :param delete_old: Whether other files with the same file extention within the directory should be deleted
    :type delete_old: bool
    :param num_max_points: The maximum number of points to be rendered
    :type num_max_points: int
    :return: The path to the PNG file saved or None if the saving failed
    :rtype: str | None
    """
    
    try:
        matplotlib.use('Agg')
        
        # Plot bars with heights from heightmap
        x_idx, y_idx = np.nonzero(heightmap)
        x = xedges[x_idx]
        y = yedges[y_idx]
        z = np.full_like(x, min_z, dtype=float)
        dx = np.full_like(x, bin_width, dtype=float)
        dy = np.full_like(y, bin_width, dtype=float)
        max_zs = heightmap[x_idx, y_idx]
        dz = max_zs - min_z

        # Use density to modulate color (optional)
        filtered_density = density[x_idx, y_idx]  
        density_normalized = (filtered_density - np.min(filtered_density)) / (np.ptp(filtered_density) + 1e-8) if np.ptp(filtered_density) > 0 else np.zeros_like(filtered_density)
        color = plt.get_cmap('viridis_r')(density_normalized)

        fig = plt.figure(layout='tight')
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(x, y, np.full_like(dz, min_z), dx, dy, dz, color)
        fig.colorbar(plt.cm.ScalarMappable(cmap='viridis_r'), ax=ax, label='Density')
        ax.set_title(f'({title.title()})')
        ax.set_xlim(x.min(), x.max() + bin_width)
        ax.set_ylim(y.min(), y.max() + bin_width)
        ax.set_zlim(min_z, max_zs.max())
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        ax.set_aspect('equal', adjustable='box')

        _, filepath = prep_dir_and_get_filepath(save_dir, prefix, suffix, "png", delete_old)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')

        plt.close(fig)

        print(f"Saved {filepath}")

        return filepath

    except Exception as e:
        traceback.print_exc()

        try:
            plt.close('all')
        except:
            pass