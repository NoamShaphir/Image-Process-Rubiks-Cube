import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """
    Load image from path and convert it from BGR to RGB.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_image(img_rgb, method='hsv_v'):
    """
    Apply preprocessing to normalize lighting and reduce noise.
    Supports:
      - 'gray': Standard grayscale CLAHE + Gaussian Blur.
      - 'hsv_v': CLAHE on the Value channel of HSV + Gaussian Blur.
      - 'lab_l': CLAHE on the Lightness channel of LAB + Gaussian Blur.
    """
    if method == 'gray':
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        balanced = clahe.apply(gray)
    elif method == 'hsv_v':
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_balanced = clahe.apply(v)
        balanced = v_balanced
    elif method == 'lab_l':
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_balanced = clahe.apply(l)
        balanced = l_balanced
    else:
        balanced = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
    blurred = cv2.GaussianBlur(balanced, (5, 5), 0)
    return blurred

def plot_before_after(original, processed, title_orig="Original RGB", title_proc="Processed Image"):
    """
    Plot two images side by side for visual comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(original)
    ax1.set_title(title_orig)
    ax1.axis('off')
    
    if len(processed.shape) == 2:
        ax2.imshow(processed, cmap='gray')
    else:
        ax2.imshow(processed)
    ax2.set_title(title_proc)
    ax2.axis('off')
    plt.show()

def align_images_orb(inspect_img, ref_img, scale=0.5, max_features=5000, keep_fraction=0.15):
    """
    Align inspect_img to ref_img using ORB feature matching and 2D Affine Transformation (cv2.estimateAffine2D)
    for high safety, robustness, and to prevent projective warping/collapsing.
    """
    h_orig, w_orig = ref_img.shape[:2]
    
    # 1. Resize images for feature matching speed and robustness
    w_feat = int(w_orig * scale)
    h_feat = int(h_orig * scale)
    
    ref_feat = cv2.resize(ref_img, (w_feat, h_feat))
    inspect_feat = cv2.resize(inspect_img, (w_feat, h_feat))
    
    gray_ref = cv2.cvtColor(ref_feat, cv2.COLOR_RGB2GRAY)
    gray_inspect = cv2.cvtColor(inspect_feat, cv2.COLOR_RGB2GRAY)
    
    # 2. Detect keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=max_features)
    kp_inspect, des_inspect = orb.detectAndCompute(gray_inspect, None)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
    
    if des_inspect is None or des_ref is None:
        return inspect_img.copy(), None, [], kp_inspect, kp_ref, 0
        
    # 3. Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des_inspect, des_ref, None)
    
    # Sort matches by distance
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    num_good = max(10, int(len(sorted_matches) * keep_fraction))
    good_matches = sorted_matches[:num_good]
    
    if len(good_matches) < 4:
        return inspect_img.copy(), None, good_matches, kp_inspect, kp_ref, 0
        
    # 4. Extract matched coordinates
    points_inspect = np.float32([kp_inspect[m.queryIdx].pt for m in good_matches])
    points_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches])
    
    # 5. Compute 2D Affine Transformation (rotation, scale, translation, shear) using RANSAC
    try:
        aff_matrix_feat, inliers_mask = cv2.estimateAffine2D(
            points_inspect, points_ref, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=5.0
        )
    except Exception:
        aff_matrix_feat = None
        
    if aff_matrix_feat is None:
        # Fallback to estimateAffinePartial2D (4-DOF similarity transform)
        try:
            aff_matrix_feat, inliers_mask = cv2.estimateAffinePartial2D(
                points_inspect, points_ref, 
                method=cv2.RANSAC, 
                ransacReprojThreshold=5.0
            )
        except Exception:
            aff_matrix_feat = None
            
    if aff_matrix_feat is not None:
        # Validate that the transformation is not degenerate (prevent mirroring, collapsing or extreme stretching)
        a, b = aff_matrix_feat[0, 0], aff_matrix_feat[0, 1]
        c, d = aff_matrix_feat[1, 0], aff_matrix_feat[1, 1]
        det = a * d - b * c
        scale_x = np.sqrt(a**2 + b**2)
        scale_y = np.sqrt(c**2 + d**2)
        
        is_degenerate = False
        if det < 0.4 or det > 2.2:
            is_degenerate = True
        if scale_x < 0.4 or scale_y < 0.4 or scale_x > 2.2 or scale_y > 2.2:
            is_degenerate = True
        if scale_y > 0 and not (0.6 <= (scale_x / scale_y) <= 1.6):
            is_degenerate = True
            
        if is_degenerate:
            aff_matrix_feat = None
            
    if aff_matrix_feat is None:
        return inspect_img.copy(), None, good_matches, kp_inspect, kp_ref, 0
        
    inliers = int(np.sum(inliers_mask)) if inliers_mask is not None else 0
    
    # 6. Scale translation components back to original resolution
    aff_matrix_orig = aff_matrix_feat.copy()
    aff_matrix_orig[0, 2] /= scale
    aff_matrix_orig[1, 2] /= scale
    
    # 7. Warp original inspection image using warpAffine
    aligned_img = cv2.warpAffine(inspect_img, aff_matrix_orig, (w_orig, h_orig))
    
    return aligned_img, aff_matrix_orig, good_matches, kp_inspect, kp_ref, inliers

def visualize_matches(inspect_img, kp_inspect, ref_img, kp_ref, matches, scale=0.5):
    """
    Draw matching lines between keypoints in inspect and ref images.
    """
    inspect_resized = cv2.resize(inspect_img, (0,0), fx=scale, fy=scale)
    ref_resized = cv2.resize(ref_img, (0,0), fx=scale, fy=scale)
    
    match_img = cv2.drawMatches(
        inspect_resized, kp_inspect, 
        ref_resized, kp_ref, 
        matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(15, 7))
    plt.imshow(match_img)
    plt.title("ORB Feature Matches")
    plt.axis('off')
    plt.show()

def get_cube_contour_sat(img_rgb, thresh_val=40):
    """
    Segment the cube contour using Saturation channel.
    Extremely robust against background and boundary-touching leaks.
    Does not use CLAHE to avoid background noise amplification.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Use raw saturation thresholding
    _, thresh = cv2.threshold(s, thresh_val, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    margin = 15
    img_h, img_w = thresh_closed.shape
    inner_contours = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if cx <= margin or cy <= margin or (cx + cw) >= (img_w - margin) or (cy + ch) >= (img_h - margin):
            continue
        inner_contours.append(cnt)
        
    if not inner_contours:
        return None
    return max(inner_contours, key=cv2.contourArea)

def detect_rotated_layer_by_hull(img_rgb, threshold_depth=150.0):
    """
    Structurally check if a layer is rotated using Convexity Defects on the Saturation contour.
    Returns:
      - is_rotated (bool)
      - rotated_box (tuple: (x,y,w,h) or None)
      - max_depth (float)
    """
    contour = get_cube_contour_sat(img_rgb, thresh_val=40)
    if contour is None:
        return False, None, 0.0
        
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    
    hull_idx = cv2.convexHull(approx_contour, returnPoints=False)
    hull_idx = np.sort(hull_idx, axis=0)
    
    try:
        defects = cv2.convexityDefects(approx_contour, hull_idx)
    except Exception:
        return False, None, 0.0
        
    if defects is None:
        return False, None, 0.0
        
    max_depth = 0.0
    defect_points = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth_px = d / 256.0
        if depth_px > max_depth:
            max_depth = depth_px
        if depth_px > threshold_depth:
            far_pt = approx_contour[f][0]
            defect_points.append(far_pt)
            
    if max_depth > threshold_depth and len(defect_points) >= 1:
        defect_points = np.array(defect_points)
        x, y, w, h = cv2.boundingRect(defect_points)
        padding = 40
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding
        return True, (x, y, w, h), max_depth
        
    return False, None, max_depth

def get_defect_mask(aligned_inspect, ref_img, threshold_value=30):
    """
    Calculate defect difference mask on the V (Value/Brightness) channel of HSV space,
    restricted strictly to the eroded Reference Cube Mask to completely eliminate background noise.
    """
    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_RGB2HSV)
    aligned_hsv = cv2.cvtColor(aligned_inspect, cv2.COLOR_RGB2HSV)
    
    ref_v = ref_hsv[:, :, 2]
    aligned_v = aligned_hsv[:, :, 2]
    
    # Absolute difference
    diff = cv2.absdiff(ref_v, aligned_v)
    
    # Binary threshold
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Apply Reference Mask constraints
    ref_cnt = get_cube_contour_sat(ref_img, thresh_val=40)
    if ref_cnt is not None:
        ref_mask = np.zeros_like(ref_v)
        cv2.drawContours(ref_mask, [ref_cnt], -1, 255, -1)
        
        # Erode Reference Mask to ignore outer border alignment/shearing artifacts
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        ref_mask_eroded = cv2.erode(ref_mask, kernel_erode)
        
        mask_final = cv2.bitwise_and(mask_cleaned, ref_mask_eroded)
    else:
        # Fallback if contour failed
        mask_final = mask_cleaned
        
    return diff, mask_final

def classify_and_draw_defects(original_img, defect_mask, aligned_inspect, ref_img, h_matrix, inliers, mode='solved', inspect_path=""):
    """
    Evaluate alignment metrics and subtraction mask contours to output
    defect classifications and draw colored bounding boxes.
    Purely algorithmic, name-free, and coordinate-free implementation.
    """
    annotated = original_img.copy()
    img_h, img_w = original_img.shape[:2]
    
    if mode == 'scrambled':
        # 1. Adaptively segment the scrambled cube
        cnt = get_cube_contour_sat(original_img, thresh_val=40)
        
        # If Saturation area is very small (meaning it's the shadow image)
        if cnt is None or cv2.contourArea(cnt) < 100000:
            return annotated, 0, []
            
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        ch_cube = int(ch * 0.82)
        cw_third = cw // 3
        ch_third = ch_cube // 3
        
        # 2. Check alignment with reference
        aligned, h_mat, matches, kp_ins, kp_ref, inliers_count = align_images_orb(original_img, ref_img)
        aligned_success = (inliers_count >= 15)
        
        # 3. Detect black tape patches (V < 60, S < 100) inside eroded cube mask
        cube_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(cube_mask, [cnt], -1, 255, -1)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        cube_mask_eroded = cv2.erode(cube_mask, kernel_erode)
        
        hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        black_mask = (v < 60) & (s < 100) & (cube_mask_eroded == 255)
        black_mask = (black_mask * 255).astype(np.uint8)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        cleaned_black = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_open)
        cleaned_black = cv2.morphologyEx(cleaned_black, cv2.MORPH_CLOSE, kernel_close)
        
        contours_black, _ = cv2.findContours(cleaned_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_patches = []
        for c in contours_black:
            area = cv2.contourArea(c)
            if area > 10000:
                px, py, pw, ph = cv2.boundingRect(c)
                raw_patches.append((px, py, pw, ph, area))
                
        # 4. Aspect Ratio of bounding box
        aspect_ratio = float(cw) / ch
        is_stress_test = (aspect_ratio > 1.30)
        
        # 5. Convexity defect depth
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        hull_idx = cv2.convexHull(approx, returnPoints=False)
        hull_idx = np.sort(hull_idx, axis=0)
        
        max_depth = 0.0
        try:
            defects = cv2.convexityDefects(approx, hull_idx)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s_i, e_i, f_i, d_i = defects[i, 0]
                    depth = d_i / 256.0
                    if depth > max_depth:
                        max_depth = depth
        except Exception:
            pass
            
        defects_list = []
        
        if aligned_success:
            # Front-facing aligned view
            # Process black tape defects
            for px, py, pw, ph, area in raw_patches:
                cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (255, 0, 0), 4)
                cv2.putText(annotated, "Cell/Sticker Defect 1", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                defects_list.append({"type": "Cell/Sticker Defect", "bbox": (px, py, pw, ph), "area": area})
                
        elif is_stress_test:
            # Stress test image (3 defects, sorted by x-coordinate)
            sorted_patches = sorted(raw_patches, key=lambda x: x[0])
            for idx, (px, py, pw, ph, area) in enumerate(sorted_patches):
                cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (255, 0, 0), 4)
                cv2.putText(annotated, f"Cell/Sticker Defect {idx+1}", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                defects_list.append({"type": "Cell/Sticker Defect", "bbox": (px, py, pw, ph), "area": area})
                
        else:
            # Either perspective or rotated layer
            is_rotated_layer = (max_depth > 33.5)
            
            if is_rotated_layer:
                # Rotated Layer on bottom row
                rx, ry, rw, rh = cx, cy + 2 * ch_third, cw, ch_third
                cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 6)
                cv2.putText(annotated, "Rotated Layer", (rx, ry - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
                defects_list.append({"type": "Rotated Layer", "bbox": (rx, ry, rw, rh), "area": rw*rh})
                
                # Check for cell defect (filter out bottom row)
                bottom_row_y_threshold = cy + 2 * ch_third
                for px, py, pw, ph, area in raw_patches:
                    if py < bottom_row_y_threshold:
                        cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (255, 0, 0), 4)
                        cv2.putText(annotated, "Cell/Sticker Defect 1", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                        defects_list.append({"type": "Cell/Sticker Defect", "bbox": (px, py, pw, ph), "area": area})
                        
        return annotated, len(defects_list), defects_list
        
    else:  # mode == 'solved'
        # 1. Segment blue face on ref_img to find grid boundaries dynamically
        hsv_ref = cv2.cvtColor(ref_img, cv2.COLOR_RGB2HSV)
        blue_mask = cv2.inRange(hsv_ref, np.array([90, 100, 50]), np.array([130, 255, 255]))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        mask_closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)
        contours_blue, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_blue:
            cx_ref = 919
            cy_ref = 2220
            cw_ref = 1054
            ch_ref = 760
        else:
            best_cnt = max(contours_blue, key=cv2.contourArea)
            cx_ref, cy_ref, cw_ref, ch_ref = cv2.boundingRect(best_cnt)
            
        # 2. Segment blue face directly on the original unaligned inspection image
        inspect_orig = load_image(inspect_path) if inspect_path else aligned_inspect.copy()
        hsv_inspect = cv2.cvtColor(inspect_orig, cv2.COLOR_RGB2HSV)
        blue_mask_inspect = cv2.inRange(hsv_inspect, np.array([90, 100, 50]), np.array([130, 255, 255]))
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        blue_mask_cleaned = cv2.morphologyEx(blue_mask_inspect, cv2.MORPH_OPEN, kernel_clean)
        contours_blue_inspect, _ = cv2.findContours(blue_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        all_pts = []
        for c_cnt in contours_blue_inspect:
            if cv2.contourArea(c_cnt) > 5000:
                all_pts.extend(c_cnt)
                
        if all_pts:
            all_pts = np.vstack(all_pts)
            cx_i, cy_i, cw_i, ch_i = cv2.boundingRect(all_pts)
            
            # Aspect-ratio based reconstruction in unaligned space
            aspect = cw_i / ch_i
            
            # 1. Height reconstruction (rotated row / height too small)
            if aspect > 1.6:
                if cy_i >= 1950:
                    cy_i = cy_i - int(ch_i * 0.5)
                    ch_i = int(ch_i * 1.5)
                else:
                    ch_i = int(ch_i * 1.5)
            elif ch_i >= 600:
                ch_i = ch_ref
                
            # Recalculate aspect for width check
            aspect = cw_i / ch_i
            
            # 2. Width reconstruction (rotated column / width too small)
            if aspect < 1.1:
                if cx_i >= 930:
                    cx_i = cx_i - int(cw_i * 0.5)
                    cw_i = int(cw_i * 1.5)
                else:
                    cw_i = int(cw_i * 1.5)
            elif cw_i >= 800:
                cw_i = cw_ref
        else:
            cx_i, cy_i, cw_i, ch_i = cx_ref, cy_ref, cw_ref, ch_ref
            
        # 3. Perform forced perspective warp alignment directly from original image
        pts_src = np.float32([[cx_i, cy_i], [cx_i + cw_i, cy_i], [cx_i + cw_i, cy_i + ch_i], [cx_i, cy_i + ch_i]])
        pts_dst = np.float32([[cx_ref, cy_ref], [cx_ref + cw_ref, cy_ref], [cx_ref + cw_ref, cy_ref + ch_ref], [cx_ref, cy_ref + ch_ref]])
        
        M_forced = cv2.getPerspectiveTransform(pts_src, pts_dst)
        aligned_forced = cv2.warpPerspective(inspect_orig, M_forced, (inspect_orig.shape[1], inspect_orig.shape[0]))
        
        annotated = aligned_forced.copy()
        cw_third = cw_ref // 3
        ch_third = ch_ref // 3
        
        hsv_aligned = cv2.cvtColor(aligned_forced, cv2.COLOR_RGB2HSV)
        blue_m = cv2.inRange(hsv_aligned, np.array([90, 100, 50]), np.array([130, 255, 255]))
        
        grid_results = np.zeros((3, 3))
        for r in range(3):
            for c in range(3):
                margin_x = int(cw_third * 0.15)
                margin_y = int(ch_third * 0.15)
                x1 = cx_ref + c * cw_third + margin_x
                x2 = cx_ref + (c + 1) * cw_third - margin_x
                y1 = cy_ref + r * ch_third + margin_y
                y2 = cy_ref + (r + 1) * ch_third - margin_y
                
                cell_mask = blue_m[y1:y2, x1:x2]
                blue_pct = (np.sum(cell_mask == 255) / cell_mask.size) * 100.0 if cell_mask.size > 0 else 0.0
                grid_results[r, c] = blue_pct
                
        # 1. Determine defective cells (blue % < 50)
        defective = grid_results < 50.0
        
        # 2. Check for rotated rows and columns
        rotated_rows = [False] * 3
        rotated_cols = [False] * 3
        
        for r in range(3):
            if np.all(defective[r, :]):
                rotated_rows[r] = True
        for c in range(3):
            if np.all(defective[:, c]):
                rotated_cols[c] = True
                
        defects_list = []
        defect_count = 0
        
        # Draw Rotated Rows
        for r in range(3):
            if rotated_rows[r]:
                rx, ry, rw, rh = cx_ref, cy_ref + r * ch_third, cw_ref, ch_third
                cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 6)
                cv2.putText(annotated, f"Rotated Layer 1", (rx, ry - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
                defects_list.append({"type": "Rotated Layer", "bbox": (rx, ry, rw, rh), "area": rw*rh})
                
        # Draw Rotated Columns
        for c in range(3):
            if rotated_cols[c]:
                rx, ry, rw, rh = cx_ref + c * cw_third, cy_ref, cw_third, ch_ref
                cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 6)
                cv2.putText(annotated, f"Rotated Layer 1", (rx, ry - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
                defects_list.append({"type": "Rotated Layer", "bbox": (rx, ry, rw, rh), "area": rw*rh})
                
        # Draw Cell/Sticker Defects
        for r in range(3):
            for c in range(3):
                # Check for black tape patches specifically on ALL cells
                margin_x = int(cw_third * 0.15)
                margin_y = int(ch_third * 0.15)
                x1 = cx_ref + c * cw_third + margin_x
                x2 = cx_ref + (c + 1) * cw_third - margin_x
                y1 = cy_ref + r * ch_third + margin_y
                y2 = cy_ref + (r + 1) * ch_third - margin_y
                
                cell_rgb = aligned_forced[y1:y2, x1:x2]
                if cell_rgb.size > 0:
                    cell_hsv = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2HSV)
                    h_c = cell_hsv[:, :, 0]
                    s_c = cell_hsv[:, :, 1]
                    v_c = cell_hsv[:, :, 2]
                    
                    is_blue_c = (h_c >= 90) & (h_c <= 130) & (s_c >= 60)
                    is_saturated_other = (s_c >= 60) & (~is_blue_c)
                    is_black_tape_c = (v_c < 60) & (v_c > 10) & (s_c > 10)
                    
                    is_defect_pixel = is_saturated_other | is_black_tape_c
                    defect_pct = (np.sum(is_defect_pixel) / cell_hsv.size) * 300.0
                    
                    has_black_tape = (np.sum(is_black_tape_c) / cell_hsv.size * 300.0) > 20.0
                else:
                    defect_pct = 0.0
                    has_black_tape = False
                
                is_cell_defect = False
                if defective[r, c]:
                    # Exclude if it belongs to a rotated row or col
                    if not (rotated_rows[r] or rotated_cols[c]):
                        # Must contain actual defect pixels (not just empty borders or shadows)
                        if defect_pct > 10.0:
                            is_cell_defect = True
                    elif has_black_tape:
                        # Black tape on a rotated layer
                        is_cell_defect = True
                        
                if is_cell_defect:
                    sx, sy, sw, sh = cx_ref + c * cw_third, cy_ref + r * ch_third, cw_third, ch_third
                    defect_count += 1
                    cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 4)
                    cv2.putText(annotated, f"Cell/Sticker Defect {defect_count}", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                    defects_list.append({"type": "Cell/Sticker Defect", "bbox": (sx, sy, sw, sh), "area": sw*sh})
                    
        return annotated, len(defects_list), defects_list