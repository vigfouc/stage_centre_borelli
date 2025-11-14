import open3d as o3d
import numpy as np
import os

def create_mesh_from_stl(stl_path, color=[0.5, 0.5, 0.5], name="Mesh"):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return {"name": name, "geometry": mesh}

def display_mesh_with_flames(stl_path, flames, camera_position=np.array([3793.47, 2113.17, 67669.8]), look_at_position=np.array([4230.49, 1462.49, 69964.6]), save_filename=None, show_window=True, save_images=False):
    
    mesh_from_stl = create_mesh_from_stl(stl_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D: Mesh, Sphere, and Axes", width=1280, height=960, visible=show_window)

    vis.add_geometry(mesh_from_stl["geometry"])
    
    if flames is not None:
        for flame in flames:
            vis.add_geometry(flame["geometry"])

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()


    # camera_position = np.array([3793.47, 2113.17, 67669.8])
    # look_at_position = np.array([4230.49, 1462.49, 69964.6])      
    # camera_position = np.array([3793.47, 2113.17, 67669.8])
    # look_at_position = np.array([3812.2803, 2098.2601, 67769.8])
    up_vector = np.array([0.0, -1.0, 0.0])

    forward = (camera_position - look_at_position)
    forward /= np.linalg.norm(forward)

    right = np.cross(up_vector, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)

    R = np.stack([right, true_up, forward], axis=1)
    t = -R.T @ camera_position

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = t

    camera_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(camera_params)

    vis.poll_events()
    vis.update_renderer()
    if save_images == True:
        output_path = os.path.join(os.getcwd(), save_filename)
        vis.capture_screen_image(output_path)
        print(f"Scene saved to: {output_path}")
        
    if show_window:
        vis.run()
    else:
        image = vis.capture_screen_float_buffer(do_render=True)
        image_array = np.asarray(image) * 255
        image_array = image_array.astype(np.uint8)
        vis.destroy_window()
        return image_array
    
def create_torus(R, r, radial_segments=100, tubular_segments=50, color=[1.0, 0.0, 0.0], name="Torus", translation=[0, 0, 0]):
    vertices = []
    triangles = []

    for i in range(radial_segments):
        theta = 2.0 * np.pi * i / radial_segments
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        for j in range(tubular_segments):
            phi = 2.0 * np.pi * j / tubular_segments
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            x = (R + r * cos_phi) * cos_theta
            y = (R + r * cos_phi) * sin_theta
            z = r * sin_phi
            vertices.append([x, y, z])

    for i in range(radial_segments):
        for j in range(tubular_segments):
            next_i = (i + 1) % radial_segments
            next_j = (j + 1) % tubular_segments

            idx0 = i * tubular_segments + j
            idx1 = next_i * tubular_segments + j
            idx2 = next_i * tubular_segments + next_j
            idx3 = i * tubular_segments + next_j

            triangles.append([idx0, idx1, idx2])
            triangles.append([idx0, idx2, idx3])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    mesh.translate(translation)
    mesh.paint_uniform_color(color)

    return {"name": name, "geometry": mesh}

if __name__ == "__main__":

    stl_path = "four_version_finale.STL"
    
    save_torus_path = "torus_in_mesh"
    os.makedirs(save_torus_path, exist_ok=True)
    
    R = 400
    r=100

    #initial_params = np.array([3917.11, 2260.72, 65871.05, 4330.44, 2106.89, 66768.55]). #good initial params

    camera_position = np.array([3917.11, 2260.72, 65871.05])
    look_at_position = np.array([4330.44, 2106.89, 66768.55]) 

    torus = create_torus(R=R, r=r, color=[0.5, 0.5, 0.5], name="TorusFlame", translation=[3083.51, 3036.26, 64129.9])
    display_mesh_with_flames(stl_path, camera_position=camera_position, look_at_position=look_at_position, flames=[], save_filename=os.path.join(save_torus_path, f"torus_{R}_{r}.png"), show_window=True, save_images=True)