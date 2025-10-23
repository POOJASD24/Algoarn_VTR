# 🧥Virtual Trial Room — Python AI-Based 3D Shirt Fitting
## 🧩Overview
This Python script enables a quick virtual try-on experience for an upper-body shirt using a combination of computer vision, AI-based depth estimation, and 3D mesh processing.
It performs a realistic shirt fitting by:
•	Capturing a live webcam frame of the user.
•	Estimating depth maps for both the person and the shirt using MiDaS.
•	Building 3D meshes from the depth data and alpha masks.
•	Detecting body keypoints using MediaPipe Pose Estimation.
•	Projecting and wrapping the shirt mesh onto the person’s body using KDTree-based surface mapping, along with scaling, rotation, and neck-hole adjustment.
•	Exporting combined OBJ/MTL files and textures to the output/ directory.

## 🧠 Libraries and Main Components
| **Library**            | **Purpose**                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------- |
| `os`                   | File and directory management (creating `output/`, handling paths).                     |
| `cv2 (OpenCV)`         | Webcam capture, image I/O, resizing, thresholding, and texture saving.                  |
| `torch`                | Loads and runs the **MiDaS** depth estimation model (GPU/CPU).                          |
| `numpy`                | Numerical computation and geometric transformations.                                    |
| `trimesh`              | Mesh loading, editing, and exporting for 3D models.                                     |
| `mediapipe`            | Pose detection (shoulders, elbows, nose, hips) and selfie segmentation for alpha masks. |
| `ctypes`               | Reads screen size to scale preview window (**Windows-specific**).                       |
| `warnings`             | Suppresses unnecessary runtime warnings.                                                |
| `scipy.spatial.KDTree` | Fast nearest-neighbor lookup to snap shirt mesh onto person mesh.                       |

## ⚙️ High-Level Pipeline (Step-by-Step)
#### 1.	Start / Setup
•	Create the output/ folder.
•	Load MiDaS small model and transformations via torch.hub.
•	Initialize MediaPipe Pose and Selfie Segmentation modules.

#### 2.	Capture
•	Open a resizable webcam preview window.
•	Press SPACE to capture the frame.

#### 3.	Foreground Mask & Person Texture
•	Generate an alpha mask using Selfie Segmentation.
•	Combine RGB and alpha channels to form an RGBA texture for the person.

#### 5.	Depth Estimation (MiDaS)
•	Run MiDaS on both the person frame and the shirt image to produce depth_person and depth_shirt.

#### 6.	Mesh Creation from Depth + Alpha
•	create_mesh_from_depth() builds:
•	Vertices (x, y, z) where z is scaled depth.
•	UV coordinates based on image-space projection.
•	Triangle faces for all valid alpha pixels.
•	Exports a simple, textured OBJ using save_obj().

#### 7.	Pose Keypoints Extraction
•	extract_pose_keypoints() returns 2D pixel coordinates for major landmarks (shoulders, elbows, wrists, hips, nose, neck).

#### 8.	Shirt Projection / Wrapping
•	project_shirt_to_upper_keypoints() performs the main wrapping process:
•	Matches 3D person vertices to 2D pose keypoints.
•	Scales the shirt mesh to match shoulder width.
•	Rotates the shirt for correct shoulder alignment.
•	Uses KDTree to “snap” shirt vertices to the nearest body surface points.
•	Extends sleeves toward elbows.
•	Cuts a neck opening based on proximity to neck center.
•	Smooths and recalculates UVs.

#### 9.	Combine & Export
•	Merges person and shirt meshes.
•	Saves final combined .obj and .mtl models with separate materials and textures inside output/.

## 💾 Files Created
•	output/person_mesh.obj + .mtl + person_texture.png
•	output/shirt_mesh.obj + .mtl + shirt_texture.png
•	output/person_with_upperbody_shirt.obj + .mtl — final fitted model

## 🧩 Key Functions 
| **Function**                                 | **Description**                                 |
| -------------------------------------------- | ----------------------------------------------- |
| `capture_image_from_webcam()`                | Opens webcam and captures a frame on key press. |
| `extract_pose_keypoints(image)`              | Returns 2D coordinates of body landmarks.       |
| `create_mesh_from_depth(depth, rgba, ...)`   | Builds 3D mesh from depth + alpha data.         |
| `save_obj(path, verts, faces, uvs, texture)` | Exports OBJ/MTL with texture references.        |
| `project_shirt_to_upper_keypoints(...)`      | Core wrapping and alignment logic.              |
| `save_combined_obj_with_shirt_wrap(...)`     | Merges meshes and saves the final model.        |

## ⚙️ Important Parameters & Heuristics
| **Parameter**                                           | **Purpose**                                             |
| ------------------------------------------------------- | ------------------------------------------------------- |
| `depth_scale = 200.0`                                   | Converts normalized MiDaS depth to Z-units.             |
| Alpha threshold = 50                                    | Filters valid mesh pixels.                              |
| Shirt scale = `(shoulder_dist * 1.18) / shirt_width_px` | Adjusts size based on body proportions.                 |
| KDTree alpha blending                                   | Controls how strongly the shirt adheres to the surface. |
| Neck radius = `0.24 * shoulder_pixel_distance`          | Defines the neck-hole size.                             |
| Laplacian smoothing                                     | Refines final mesh for smoother appearance.             |

## ⚠️Limitations & Issues
#### 🔹 Depth / Geometry
•	MiDaS produces relative depth, not metric — scaling can vary.
•	No camera intrinsics, so 3D positions are approximate.
•	Meshes may be dense or noisy, not ideal for physics or animation.
•	Normals not exported — shading may look inconsistent.

#### 🔹 Wrapping / Alignment
•	KDTree snapping is naïve — can clip through body.
•	No physics (gravity, drape, collisions).
•	Neck-hole carving is 2D heuristic, may over/under-cut.
•	Sleeve alignment is rough, not topology-aware.
•	Rotation limited to Y-axis only — torso tilt may misalign.

#### 🔹 Pose & Landmarks
•	Uses static MediaPipe mode — no temporal smoothing.
•	Neck point is an estimated average, may vary across poses.

#### 🔹 Texturing / UV Mapping
•	Bounding-box UV projection — may stretch shirt textures.
•	Minimal material separation and surface color settings.

#### 🔹 Reliability / Compatibility
•	Windows-specific (uses ctypes.windll.user32).
•	Needs internet on first run to download MiDaS model.
•	High memory usage for large meshes.
•	Unity/Three.js may require rotation/scaling fix.

## 🚀 Potential Improvements
•	Integrate camera intrinsics for real 3D coordinate reprojection.
•	Use normal-aware projection for better surface conformity.
•	Replace KDTree with closest-point-on-triangle mapping.
•	Add cloth simulation (mass-spring or physics-based).
•	Detect neck hole in 3D instead of 2D projection.
•	Perform retopology for clean, manifold meshes.
•	Preserve original shirt UVs via UV-transfer.
•	Extend to MediaPipe Holistic for full-body keypoints.
•	Add temporal smoothing for real-time webcam mode.

## 📦 Runtime Notes & Requirements
•	Requires E:\VTR PROJECT\output\blackshirt.jpg (shirt image).
•	First run downloads MiDaS model via torch.hub.
•	Outputs saved to output/ directory.

## 🧪 Common Runtime Issues
| **Issue**                       | **Description**                              |
| ------------------------------- | -------------------------------------------- |
| Shirt floating or clipping      | Incorrect depth scaling or alignment.        |
| Neck hole too large / misplaced | Neck heuristic mismatch.                     |
| Sleeves look distorted          | Elbow projection not aligned.                |
| OBJ imports incorrectly         | Axis mismatch or missing normals.            |
| Slow performance                | High-res frames or CPU-only MiDaS inference. |

## ✅ Result:
Generates a realistic 3D shirt fitting mesh aligned to the user’s pose and anatomy, exportable to Unity, Blender, or Three.js for visualization and further processing.
	


