# vtr_wrap_full.py
import os
import cv2
import torch
import numpy as np
import trimesh
import mediapipe as mp
import ctypes
import warnings
from scipy.spatial import KDTree

warnings.filterwarnings("ignore")

# ----------------------------
# MediaPipe (pose + selfie seg)
# ----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
mp_seg = mp.solutions.selfie_segmentation

# ----------------------------
# Utility: webcam capture
# ----------------------------
def capture_image_from_webcam(window_title="Webcam - Press SPACE to capture"):
    user32 = ctypes.windll.user32
    screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, int(screen_w * 0.6), int(screen_h * 0.75))
    print("Press SPACE to capture frame or ESC to quit.")
    frame = None
    while True:
        ret, frm = cap.read()
        if not ret:
            print("Webcam read failed.")
            break
        cv2.imshow(window_title, frm)
        k = cv2.waitKey(1)
        if k == 32:
            frame = frm.copy()
            print("Captured frame.")
            break
        elif k == 27:
            print("Cancelled by user.")
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

# ----------------------------
# Pose extraction (2D)
# ----------------------------
def extract_pose_keypoints(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if not results.pose_landmarks:
        return None
    L = mp_pose.PoseLandmark
    h, w = image.shape[:2]
    def xy(lm): return (int(lm.x * w), int(lm.y * h))
    lm = results.pose_landmarks.landmark
    kpts = {}
    keys = [
        ("left_shoulder", L.LEFT_SHOULDER), ("right_shoulder", L.RIGHT_SHOULDER),
        ("left_elbow", L.LEFT_ELBOW), ("right_elbow", L.RIGHT_ELBOW),
        ("left_wrist", L.LEFT_WRIST), ("right_wrist", L.RIGHT_WRIST),
        ("left_hip", L.LEFT_HIP), ("right_hip", L.RIGHT_HIP),
        ("nose", L.NOSE)
    ]
    for name, idx in keys:
        kpts[name] = xy(lm[idx])
    mid_sh = ((kpts["left_shoulder"][0] + kpts["right_shoulder"][0]) // 2,
              (kpts["left_shoulder"][1] + kpts["right_shoulder"][1]) // 2)
    # neck slightly below shoulder-center toward nose
    kpts["neck"] = (mid_sh[0], int(mid_sh[1] + 0.12 * abs(mid_sh[1] - kpts["nose"][1])))
    return kpts

# ----------------------------
# Helpers: nearest 2D -> 3D mapping
# ----------------------------
def find_nearest_vertex_2d(kpt_2d, vertices_3d):
    if vertices_3d.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    verts2d = vertices_3d[:, :2]
    d = np.linalg.norm(verts2d - np.array(kpt_2d), axis=1)
    return vertices_3d[int(np.argmin(d))]

# ----------------------------
# Laplacian smoothing
# ----------------------------
def laplacian_smooth(vertices, faces, lamb=0.35, iterations=15):
    V = vertices.copy()
    n = len(V)
    adj = [[] for _ in range(n)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        if b not in adj[a]: adj[a].append(b)
        if c not in adj[a]: adj[a].append(c)
        if a not in adj[b]: adj[b].append(a)
        if c not in adj[b]: adj[b].append(c)
        if a not in adj[c]: adj[c].append(a)
        if b not in adj[c]: adj[c].append(b)
    for _ in range(iterations):
        V_new = V.copy()
        for i, nbrs in enumerate(adj):
            if not nbrs: continue
            V_new[i] = V[i] + lamb * (np.mean(V[nbrs], axis=0) - V[i])
        V = V_new
    return V

# ----------------------------
# Create mesh from depth and RGBA mask
# ----------------------------
def create_mesh_from_depth(depth, rgba, alpha_threshold=50, depth_scale=200.0):
    h, w = depth.shape
    idx_map = -np.ones((h, w), dtype=int)
    vertices = []
    uvs = []
    # normalize depth to [0,1]
    dmin, dmax = depth.min(), depth.max()
    if (dmax - dmin) < 1e-6:
        norm = np.zeros_like(depth)
    else:
        norm = (depth - dmin) / (dmax - dmin)
    zmap = norm * depth_scale
    idx = 0
    for y in range(h):
        for x in range(w):
            if rgba[y, x, 3] > alpha_threshold:
                vertices.append((float(x), float(y), float(zmap[y, x])))
                uvs.append((x / (w - 1), 1.0 - y / (h - 1)))
                idx_map[y, x] = idx
                idx += 1
    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            a = idx_map[y, x]; b = idx_map[y + 1, x]
            c = idx_map[y + 1, x + 1]; d = idx_map[y, x + 1]
            if a != -1 and b != -1 and c != -1:
                faces.append((a, b, c))
            if a != -1 and c != -1 and d != -1:
                faces.append((a, c, d))
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32), np.array(uvs, dtype=np.float32)

# ----------------------------
# Save simple OBJ (vertex & vt indices aligned)
# ----------------------------
def save_obj(path, verts, faces, uvs, texture_filename):
    mtl = path.replace(".obj", ".mtl")
    with open(path, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl)}\n")
        f.write("usemtl m1\n")
        for v in verts:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(float(v[0]), float(v[1]), float(v[2])))
        for uv in uvs:
            f.write("vt {:.6f} {:.6f}\n".format(float(uv[0]), float(uv[1])))
        for face in faces:
            a, b, c = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    with open(mtl, "w") as m:
        m.write("newmtl m1\n")
        m.write(f"map_Kd {os.path.basename(texture_filename)}\n")

# ----------------------------
# Core: project + wrap shirt
# ----------------------------
def project_shirt_to_upper_keypoints(shirt_vertices, person_vertices, keypoints_2d, shirt_faces=None):
    # If empty inputs, return gracefully
    if shirt_vertices.size == 0 or person_vertices.size == 0:
        return shirt_vertices, (shirt_faces if shirt_faces is not None else np.array([], dtype=np.int32)), np.zeros((0,2), dtype=np.float32)

    # anchors (3D)
    left3d = find_nearest_vertex_2d(keypoints_2d["left_shoulder"], person_vertices)
    right3d = find_nearest_vertex_2d(keypoints_2d["right_shoulder"], person_vertices)
    left_elbow3d = find_nearest_vertex_2d(keypoints_2d["left_elbow"], person_vertices)
    right_elbow3d = find_nearest_vertex_2d(keypoints_2d["right_elbow"], person_vertices)
    neck3d = find_nearest_vertex_2d(keypoints_2d["neck"], person_vertices)

    # chest anchor (below shoulders)
    shoulder_center_2d = np.mean([keypoints_2d["left_shoulder"], keypoints_2d["right_shoulder"]], axis=0)
    chest2d_y = shoulder_center_2d[1] + 0.48 * abs(keypoints_2d["left_shoulder"][1] - keypoints_2d["right_shoulder"][1])
    chest3d = find_nearest_vertex_2d((shoulder_center_2d[0], chest2d_y), person_vertices)

    # scale shirt by shoulder distance
    shoulder_dist = np.linalg.norm(right3d - left3d) + 1e-6
    shirt_width_px = max(1e-6, np.max(shirt_vertices[:, 0]) - np.min(shirt_vertices[:, 0]))
    scale_factor = (shoulder_dist * 1.18) / shirt_width_px
    s_center = np.mean(shirt_vertices, axis=0)
    scaled = (shirt_vertices - s_center) * scale_factor

    # rotate to align shoulders (x,z plane)
    shoulder_vec = right3d - left3d
    angle = np.arctan2(shoulder_vec[2], shoulder_vec[0])
    R = np.array([[np.cos(angle), 0.0, -np.sin(angle)],
                  [0.0, 1.0, 0.0],
                  [np.sin(angle), 0.0,  np.cos(angle)]], dtype=np.float32)
    rotated = scaled @ R.T

    # translate chest centers together
    translated = rotated + (chest3d - np.mean(rotated, axis=0))

    # KDTree for person surface
    kd = KDTree(person_vertices)

    wrapped = np.zeros_like(translated)
    tx_mean_x = np.mean(translated[:,0])
    tx_range_x = np.max(translated[:,0]) - np.min(translated[:,0]) + 1e-6

    for i, v in enumerate(translated):
        dist, idx = kd.query(v)
        target = person_vertices[idx]
        alpha = float(min(1.0, dist / (0.06 * shoulder_dist + 1e-6)))
        x_rel = abs(v[0] - tx_mean_x)
        edge_factor = np.clip(x_rel / tx_range_x, 0.0, 1.0)
        # Sleeve pull
        if edge_factor > 0.6:
            elbow_target = left_elbow3d if v[0] < tx_mean_x else right_elbow3d
            wrapped_v = (1 - alpha) * target + alpha * (0.78 * v + 0.22 * elbow_target)
        else:
            wrapped_v = (1 - alpha) * target + alpha * v
        wrapped[i] = wrapped_v

    # carve neck hole in 2D XY pixel space
    neck_center_2d = np.array(keypoints_2d["neck"])
    neck_radius_px = 0.24 * np.linalg.norm(np.array(keypoints_2d["left_shoulder"]) - np.array(keypoints_2d["right_shoulder"]))
    d2neck = np.linalg.norm(wrapped[:, :2] - neck_center_2d, axis=1)
    keep = d2neck > neck_radius_px
    if np.sum(keep) < max(3, len(keep)//10):  # ensure not removing too much
        keep[:] = True
    wrapped_kept = wrapped[keep]

    # remap faces if given
    new_faces = np.array([], dtype=np.int32)
    if shirt_faces is not None and shirt_faces.size > 0:
        old_to_new = -np.ones(len(wrapped), dtype=int)
        old_to_new[np.where(keep)[0]] = np.arange(np.sum(keep), dtype=int)
        faces_new_list = []
        for face in shirt_faces:
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            if old_to_new[a] != -1 and old_to_new[b] != -1 and old_to_new[c] != -1:
                faces_new_list.append((old_to_new[a], old_to_new[b], old_to_new[c]))
        if faces_new_list:
            new_faces = np.array(faces_new_list, dtype=np.int32)

    # smoothing
    if new_faces.size != 0 and wrapped_kept.shape[0] > 3:
        wrapped_kept = laplacian_smooth(wrapped_kept, new_faces, lamb=0.32, iterations=18)

    # recompute UVs by bounding-box projection
    if wrapped_kept.shape[0] > 0:
        min_x, max_x = wrapped_kept[:,0].min(), wrapped_kept[:,0].max()
        min_y, max_y = wrapped_kept[:,1].min(), wrapped_kept[:,1].max()
        dx = max_x - min_x if (max_x - min_x) > 1e-6 else 1.0
        dy = max_y - min_y if (max_y - min_y) > 1e-6 else 1.0
        new_uvs = np.stack([(wrapped_kept[:,0] - min_x) / dx, 1.0 - (wrapped_kept[:,1] - min_y) / dy], axis=1)
    else:
        new_uvs = np.zeros((0,2), dtype=np.float32)

    return wrapped_kept, new_faces, new_uvs

# ----------------------------
# Export combined person + shirt
# ----------------------------
def save_combined_obj_with_shirt_wrap(person_path, shirt_path, output_path, ref_image):
    person = trimesh.load(person_path, force='mesh', process=False)
    shirt = trimesh.load(shirt_path, force='mesh', process=False)
    keypoints = extract_pose_keypoints(ref_image)
    if keypoints is None:
        print("No pose detected. Aborting wrap.")
        return
    p_vertices = np.array(person.vertices, dtype=np.float32)
    p_faces = np.array(person.faces, dtype=np.int32) if hasattr(person, 'faces') else np.array([], dtype=np.int32)
    try:
        p_uv = np.array(person.visual.uv, dtype=np.float32) if person.visual.uv is not None else np.zeros((len(p_vertices),2), dtype=np.float32)
    except Exception:
        p_uv = np.zeros((len(p_vertices),2), dtype=np.float32)

    s_vertices = np.array(shirt.vertices, dtype=np.float32)
    s_faces = np.array(shirt.faces, dtype=np.int32) if hasattr(shirt, 'faces') else np.array([], dtype=np.int32)

    wrapped, new_faces, new_uvs = project_shirt_to_upper_keypoints(s_vertices, p_vertices, keypoints, shirt_faces=s_faces)

    # combine
    if wrapped.size == 0:
        print("Wrapped shirt empty -> skipping merge.")
        return
    combined_vertices = np.vstack((p_vertices, wrapped))
    combined_uvs = np.vstack((p_uv, new_uvs)) if new_uvs.size != 0 else p_uv
    # faces
    if p_faces.size != 0:
        if new_faces.size == 0:
            combined_faces = p_faces
        else:
            shifted = new_faces + len(p_vertices)
            combined_faces = np.vstack((p_faces, shifted))
    else:
        combined_faces = new_faces + len(p_vertices)

    # write mtl
    mtl_path = output_path.replace(".obj", ".mtl")
    with open(mtl_path, "w") as m:
        m.write("newmtl person_mat\nmap_Kd person_texture.png\n")
        m.write("newmtl shirt_mat\nmap_Kd shirt_texture.png\n")

    # write obj (person faces first -> person_mat, then shirt_mat)
    with open(output_path, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        # vertices
        for v in combined_vertices:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(float(v[0]), float(v[1]), float(v[2])))
        # uvs
        for uv in combined_uvs:
            f.write("vt {:.6f} {:.6f}\n".format(float(uv[0]), float(uv[1])))
        # person faces
        f.write("usemtl person_mat\n")
        if p_faces.size != 0:
            for face in p_faces:
                a, b, c = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
                f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
        # shirt faces
        if new_faces.size != 0:
            f.write("usemtl shirt_mat\n")
            for face in (new_faces + len(p_vertices)):
                a, b, c = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
                f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    print(f"Saved combined OBJ: {output_path}")

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    os.makedirs("output", exist_ok=True)

    # load MiDaS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading MiDaS (may take a few seconds)...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    # capture a frame
    frame = capture_image_from_webcam()
    if frame is None:
        print("No frame captured. Exiting.")
        return

    # foreground mask
    seg = mp_seg.SelfieSegmentation(model_selection=1)
    mask = seg.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask > 0.5
    alpha_mask = (mask * 255).astype(np.uint8)
    rgba_person = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    rgba_person[:, :, 3] = alpha_mask

    # MiDaS depth for person
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = transform(rgb).to(device)
    with torch.no_grad():
        pred = midas(t)
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False).squeeze()
    depth_person = pred.cpu().numpy()

    # build person mesh
    person_verts, person_faces, person_uvs = create_mesh_from_depth(depth_person, rgba_person)
    person_obj = "output/person_mesh.obj"
    person_tex = "output/person_texture.png"
    cv2.imwrite(person_tex, rgba_person)
    save_obj(person_obj, person_verts, person_faces, person_uvs, person_tex)
    print("Saved person mesh + texture.")

    # load shirt image
    shirt_path = "output/blackshirt.jpg"
    if not os.path.exists(shirt_path):
        print("Shirt image not found at output/blackshirt.jpg. Place it there and re-run.")
        return
    shirt_img = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
    if shirt_img is None:
        print("Failed to read shirt image. Exiting.")
        return
    # resize shirt to person image width
    target_w = rgba_person.shape[1]
    sh = int(shirt_img.shape[0] * (target_w / max(1, shirt_img.shape[1])))
    resized = cv2.resize(shirt_img, (target_w, sh))
    if resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        alpha_s = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1]
    else:
        alpha_s = resized[:, :, 3]
    rgba_shirt = cv2.cvtColor(resized[:, :, :3], cv2.COLOR_BGR2BGRA)
    rgba_shirt[:, :, 3] = alpha_s

    # MiDaS depth for shirt
    with torch.no_grad():
        ts = transform(cv2.cvtColor(resized[:, :, :3], cv2.COLOR_BGR2RGB)).to(device)
        ps = midas(ts)
        ps = torch.nn.functional.interpolate(ps.unsqueeze(1), size=resized.shape[:2], mode="bicubic", align_corners=False).squeeze()
    depth_shirt = ps.cpu().numpy()

    shirt_verts, shirt_faces, shirt_uvs = create_mesh_from_depth(depth_shirt, rgba_shirt)
    shirt_obj = "output/shirt_mesh.obj"
    shirt_tex = "output/shirt_texture.png"
    cv2.imwrite(shirt_tex, rgba_shirt)
    save_obj(shirt_obj, shirt_verts, shirt_faces, shirt_uvs, shirt_tex)
    print("Saved shirt mesh + texture.")

    # combine & wrap
    out_obj = "output/person_with_upperbody_shirt.obj"
    save_combined_obj_with_shirt_wrap(person_obj, shirt_obj, out_obj, frame)
    print("Main pipeline complete. Check output/ folder.")

if __name__ == "__main__":
    main()
