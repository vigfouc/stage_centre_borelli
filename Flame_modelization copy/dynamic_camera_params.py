import open3d as o3d
import numpy as np
import cv2


def create_mesh_from_stl(stl_path, color=[0.5, 0.5, 0.5], name="Mesh"):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return {"name": name, "geometry": mesh}


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


class InteractiveCameraTuner:
    def __init__(self, stl_path, flames=None, initial_camera_pos=None, initial_look_at=None):
        self.stl_path = stl_path
        self.flames = flames if flames is not None else []

        # Initial parameters
        self.camera_position = initial_camera_pos if initial_camera_pos is not None else np.array([3793.47, 2113.17, 67669.8])
        self.look_at_position = initial_look_at if initial_look_at is not None else np.array([4230.49, 1462.49, 69964.6])

        # Step sizes
        self.cam_step = 50.0
        self.look_step = 50.0
        self.cam_step_fine = 10.0
        self.look_step_fine = 10.0
        self.fine_mode = False

        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Interactive Camera Tuner", width=1280, height=960)

        # Load mesh
        mesh = create_mesh_from_stl(stl_path)
        self.vis.add_geometry(mesh["geometry"])

        # Add flames if any
        for flame in self.flames:
            self.vis.add_geometry(flame["geometry"])

        # Register key callbacks
        self.register_callbacks()

        # Update camera
        self.update_camera()

        # Print help
        self.print_instructions()

    def print_instructions(self):
        print("\n" + "=" * 70)
        print("INTERACTIVE CAMERA PARAMETER TUNER")
        print("=" * 70)
        print("\nCURRENT PARAMETERS:")
        print(f"Camera Position: {self.camera_position}")
        print(f"Look-At Position: {self.look_at_position}")
        print("\n" + "-" * 70)
        print("KEYBOARD CONTROLS:")
        print("-" * 70)
        print("\n📷 CAMERA POSITION (Arrow Keys + Page Up/Down):")
        print("  ← → : Move X axis (left/right)")
        print("  ↑ ↓ : Move Y axis (up/down)")
        print("  PgUp/PgDn : Move Z axis (forward/back)")
        print("\n🎯 LOOK-AT POSITION (WASD + Q/E):")
        print("  A D : Move X axis (left/right)")
        print("  W S : Move Y axis (up/down)")
        print("  Q E : Move Z axis (forward/back)")
        print("\n⚙️  CONTROLS:")
        print("  F : Toggle Fine adjustment (10x smaller steps)")
        print("  P : Print current parameters")
        print("  C : Copy parameters to clipboard")
        print("  R : Reset to initial parameters")
        print("  O : Show reference overlay")
        print("  H : Show this help")
        print("  ESC : Exit")
        print("=" * 70 + "\n")

    def update_camera(self):
        ctr = self.vis.get_view_control()
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = ctr.convert_to_pinhole_camera_parameters().intrinsic

        forward = (self.look_at_position - self.camera_position)
        forward /= np.linalg.norm(forward)
        up_vector = np.array([0.0, -1.0, 0.0])
        right = np.cross(forward, up_vector)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)

        R = np.stack([right, true_up, -forward], axis=1)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R.T
        extrinsic[:3, 3] = -R.T @ self.camera_position
        params.extrinsic = extrinsic

        ctr.convert_from_pinhole_camera_parameters(params)
        self.vis.poll_events()
        self.vis.update_renderer()

    def get_step_size(self, fine):
        return self.cam_step_fine if fine else self.cam_step

    def get_look_step_size(self, fine):
        return self.look_step_fine if fine else self.look_step

    def register_callbacks(self):
        # Camera
        self.vis.register_key_callback(262, lambda vis: self.adjust_camera(0, 1))  # →
        self.vis.register_key_callback(263, lambda vis: self.adjust_camera(0, -1))  # ←
        self.vis.register_key_callback(265, lambda vis: self.adjust_camera(1, 1))  # ↑
        self.vis.register_key_callback(264, lambda vis: self.adjust_camera(1, -1))  # ↓
        self.vis.register_key_callback(266, lambda vis: self.adjust_camera(2, 1))  # PgUp
        self.vis.register_key_callback(267, lambda vis: self.adjust_camera(2, -1))  # PgDn

        # Look-at
        self.vis.register_key_callback(ord('D'), lambda vis: self.adjust_look_at(0, 1))
        self.vis.register_key_callback(ord('A'), lambda vis: self.adjust_look_at(0, -1))
        self.vis.register_key_callback(ord('W'), lambda vis: self.adjust_look_at(1, 1))
        self.vis.register_key_callback(ord('S'), lambda vis: self.adjust_look_at(1, -1))
        self.vis.register_key_callback(ord('E'), lambda vis: self.adjust_look_at(2, 1))
        self.vis.register_key_callback(ord('Q'), lambda vis: self.adjust_look_at(2, -1))

        # Utilities
        self.vis.register_key_callback(ord('F'), lambda vis: self.toggle_fine_mode())
        self.vis.register_key_callback(ord('P'), lambda vis: self.print_parameters())
        self.vis.register_key_callback(ord('C'), lambda vis: self.copy_parameters())
        self.vis.register_key_callback(ord('R'), lambda vis: self.reset_parameters())
        self.vis.register_key_callback(ord('H'), lambda vis: self.print_instructions())
        self.vis.register_key_callback(ord('O'), lambda vis: self.show_overlay("std_image.png"))

    def adjust_camera(self, axis, direction):
        step = self.get_step_size(self.fine_mode)
        self.camera_position[axis] += direction * step
        self.update_camera()
        print(f"Camera {['X','Y','Z'][axis]}: {self.camera_position[axis]:.2f}")

    def adjust_look_at(self, axis, direction):
        step = self.get_look_step_size(self.fine_mode)
        self.look_at_position[axis] += direction * step
        self.update_camera()
        print(f"Look-At {['X','Y','Z'][axis]}: {self.look_at_position[axis]:.2f}")

    def toggle_fine_mode(self):
        self.fine_mode = not self.fine_mode
        mode = "FINE" if self.fine_mode else "NORMAL"
        print(f"\n🔧 Adjustment mode: {mode}\n")

    def print_parameters(self):
        print(f"camera_position = {self.camera_position.tolist()}")
        print(f"look_at_position = {self.look_at_position.tolist()}")

    def copy_parameters(self):
        params = f"camera_position = {self.camera_position.tolist()}\nlook_at_position = {self.look_at_position.tolist()}"
        try:
            import pyperclip
            pyperclip.copy(params)
            print("✓ Parameters copied to clipboard!")
        except ImportError:
            print("⚠ pyperclip not installed.")
            self.print_parameters()

    def reset_parameters(self):
        self.camera_position = np.array([3793.47, 2113.17, 67669.8])
        self.look_at_position = np.array([4230.49, 1462.49, 69964.6])
        self.update_camera()
        print("🔄 Reset to initial parameters")
        self.print_parameters()

    def show_overlay(self, reference_image_path, alpha=0.4):
        """Display a transparent overlay of reference image"""
        ref_img = cv2.imread(reference_image_path)
        if ref_img is None:
            print(f"⚠ Could not load reference image: {reference_image_path}")
            return
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.resize(ref_img, (1280, 960))

        print("🖼 Overlay mode ON. Close overlay with ESC.")

        while True:
            img_o3d = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
            img_o3d = (img_o3d * 255).astype(np.uint8)
            img_o3d = cv2.cvtColor(img_o3d, cv2.COLOR_RGB2BGR)

            ref_resized = cv2.resize(ref_img, (img_o3d.shape[1], img_o3d.shape[0]))
            overlay = cv2.addWeighted(ref_resized, alpha, img_o3d, 1 - alpha, 0)

            cv2.imshow("Overlay: model vs. reference", overlay)
            key = cv2.waitKey(200) & 0xFF
            if key == 27:  # ESC
                break
        cv2.destroyAllWindows()

    def run(self):
        self.vis.run()

        # Extract real final camera from Open3D
        ctr = self.vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        self.vis.destroy_window()

        extrinsic = np.linalg.inv(params.extrinsic)
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]

        self.camera_position = t
        forward = -R[:,2]
        self.look_at_position = t + forward * 1000.0

        print("FINAL CAMERA PARAMETERS:")
        print(f"camera_position = {self.camera_position.tolist()}")
        print(f"look_at_position = {self.look_at_position.tolist()}")

        return self.camera_position, self.look_at_position


if __name__ == "__main__":
    stl_path = "four_version_finale.STL"

    # Optional torus flame
    R = 400
    r = 100
    torus = create_torus(R=R, r=r, color=[0.5,0.5,0.5], translation=[3083.51,3036.26,64129.9])

    # Initial camera

    #initial_camera = np.array([3525.20, 2027.26, 67277.01])
    #initial_look_at = np.array([4200.07, 1558.07, 70268.76])
    
    initial_camera = np.array([3917.11, 2260.72, 65871.05])
    initial_look_at = np.array([4330.44, 2106.89, 66768.55])

    tuner = InteractiveCameraTuner(
        stl_path=stl_path,
        flames=[],
        initial_camera_pos=initial_camera,
        initial_look_at=initial_look_at
    )

    final_camera, final_look_at = tuner.run()

    print(f"\nUse these parameters in your optimization:")
    print(f"initial_params = np.array([{final_camera[0]:.2f}, {final_camera[1]:.2f}, {final_camera[2]:.2f}, "
          f"{final_look_at[0]:.2f}, {final_look_at[1]:.2f}, {final_look_at[2]:.2f}])")
