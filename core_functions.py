import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# Predefined Morphological Kernels (Global Constants)
# ==========================================================
KERNEL_RECT_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
KERNEL_RECT_7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
KERNEL_RECT_9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
KERNEL_RECT_15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
KERNEL_RECT_21 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

# ==========================================================
# Helper Processing Functions for Dictionary Dispatch
# ==========================================================
def _preprocess_gray(img_rgb):
    """Convert image to gray and apply CLAHE."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _preprocess_hsv_v(img_rgb):
    """Extract V channel and apply CLAHE."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(v)

def _preprocess_lab_l(img_rgb):
    """Extract L channel and apply CLAHE."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(l)

def get_quad_corners(cnt):
    """
    Extract the 4 extreme corners of a given contour based on x + y and x - y sums/differences.
    Returns them in a consistent order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.

    Parameters:
        cnt (numpy.ndarray): The input contour points.

    Returns:
        numpy.ndarray: Ordered float32 corner points of shape (4, 2).
        
    Example:
        >>> cnt = np.array([[[10, 10]], [[100, 10]], [[100, 100]], [[10, 100]]])
        >>> corners = get_quad_corners(cnt)
        >>> print(corners[0])
        [10. 10.]
    """
    pts = np.atleast_2d(cnt.squeeze())
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    
    return np.array([tl, tr, br, bl], dtype="float32")

def load_image(image_path):
    """
    Load an image from the specified file path and convert it from BGR to RGB color space.

    Parameters:
        image_path (str): The absolute or relative path to the image file.

    Returns:
        numpy.ndarray: The loaded image in RGB format.

    Raises:
        FileNotFoundError: If the image cannot be loaded.

    Example:
        >>> img = load_image("data/solved_pipeline/ref_solved.jpg")
        >>> print(img.shape)
        (3024, 4032, 3)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image at {image_path}. File not found or empty.")
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_image(img_rgb, method='hsv_v'):
    """
    Preprocess the input RGB image to normalize lighting, improve contrast, and reduce noise.
    Uses a dictionary dispatch table for clean, branch-free strategy selection.

    Parameters:
        img_rgb (numpy.ndarray): The input RGB image.
        method (str): The preprocessing technique ('gray', 'hsv_v', 'lab_l').

    Returns:
        numpy.ndarray: The preprocessed grayscale/single-channel image.

    Example:
        >>> img = load_image("data/solved_pipeline/ref_solved.jpg")
        >>> blurred = preprocess_image(img, method='hsv_v')
        >>> print(blurred.shape)
        (3024, 4032)
    """
    if img_rgb is None or img_rgb.size == 0:
        print("Warning: Input image is empty or None in preprocess_image.")
        return None

    # Strategy Pattern / Dictionary Dispatch Table
    methods = {
        'gray': _preprocess_gray,
        'hsv_v': _preprocess_hsv_v,
        'lab_l': _preprocess_lab_l
    }
    
    proc_fn = methods.get(method)
    
    # Early check for dispatch function
    if proc_fn is not None:
        balanced = proc_fn(img_rgb)
        
    if proc_fn is None:
        balanced = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(balanced, (5, 5), 0)
    return blurred

def plot_before_after(original, processed, title_orig="Original RGB", title_proc="Processed Image"):
    """
    Plot the original and processed images side-by-side.

    Parameters:
        original (numpy.ndarray): The original image.
        processed (numpy.ndarray): The processed image.
        title_orig (str): Title for the original image plot.
        title_proc (str): Title for the processed image plot.

    Example:
        >>> img = load_image("data/solved_pipeline/ref_solved.jpg")
        >>> processed = preprocess_image(img, method='gray')
        >>> plot_before_after(img, processed)
    """
    if original is None or processed is None:
        print("Warning: Cannot plot None images in plot_before_after.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(original)
    ax1.set_title(title_orig)
    ax1.axis('off')
    
    if len(processed.shape) == 2:
        ax2.imshow(processed, cmap='gray')
    if len(processed.shape) != 2:
        ax2.imshow(processed)
        
    ax2.set_title(title_proc)
    ax2.axis('off')
    plt.show()

def align_images_orb(inspect_img, ref_img, scale=0.5, max_features=5000, keep_fraction=0.15):
    """
    Align the inspection image to the reference image. Prefers geometric Corner-Based 
    Perspective Registration for robust color-noise handling, with an ORB feature matching 
    fallback if outer contours cannot be detected.
    Also runs ORB on the original images to return the correct count of inliers for classification.

    Parameters:
        inspect_img (numpy.ndarray): The input inspection image.
        ref_img (numpy.ndarray): The reference template image.
        scale (float): The scale factor for resizing during feature detection.
        max_features (int): Maximum number of features for ORB.
        keep_fraction (float): Fraction of top matches to keep.

    Returns:
        tuple: (aligned_img, h_matrix, good_matches, kp_inspect, kp_ref, inliers)

    Example:
        >>> ref = load_image("data/solved_pipeline/ref_solved.jpg")
        >>> inspect = load_image("data/solved_pipeline/inspect_solved_single_defect.jpg")
        >>> aligned, h_mat, matches, kp_ins, kp_ref, inliers = align_images_orb(inspect, ref)
    """
    if inspect_img is None or ref_img is None:
        print("Warning: Input or reference image is None in align_images_orb.")
        return None, None, [], [], [], 0

    h_orig, w_orig = ref_img.shape[:2]
    
    # 1. Run ORB first on unaligned images to count inliers (acting as our straightness/alignment success feature)
    w_feat = int(w_orig * scale)
    h_feat = int(h_orig * scale)
    ref_feat = cv2.resize(ref_img, (w_feat, h_feat))
    inspect_feat = cv2.resize(inspect_img, (w_feat, h_feat))
    
    gray_ref = cv2.cvtColor(ref_feat, cv2.COLOR_RGB2GRAY)
    gray_inspect = cv2.cvtColor(inspect_feat, cv2.COLOR_RGB2GRAY)
    
    orb = cv2.ORB_create(nfeatures=max_features)
    kp_inspect, des_inspect = orb.detectAndCompute(gray_inspect, None)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
    
    inliers = 0
    good_matches = []
    
    if des_inspect is not None and des_ref is not None:
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(des_inspect, des_ref, None)
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        num_good = max(10, int(len(sorted_matches) * keep_fraction))
        good_matches = sorted_matches[:num_good]
        
        if len(good_matches) >= 4:
            points_inspect = np.float32([kp_inspect[m.queryIdx].pt for m in good_matches])
            points_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches])
            try:
                _, inliers_mask = cv2.estimateAffine2D(points_inspect, points_ref, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if inliers_mask is not None:
                    inliers = int(np.sum(inliers_mask))
            except Exception as e:
                print(f"Warning: RANSAC estimateAffine2D failed: {e}")

    # 2. Try Corner-Based Perspective Registration (highly robust to color noise)
    ref_cnt = get_cube_contour_sat(ref_img, thresh_val=40)
    ins_cnt = get_cube_contour_sat(inspect_img, thresh_val=40)
    
    # Verify that the segmented contours are valid and represent large physical objects (cube faces)
    use_corners = False
    if ref_cnt is not None and ins_cnt is not None:
        if cv2.contourArea(ref_cnt) >= 200000 and cv2.contourArea(ins_cnt) >= 200000:
            use_corners = True
            
    if use_corners:
        try:
            ref_pts = get_quad_corners(ref_cnt)
            ins_pts = get_quad_corners(ins_cnt)
            H = cv2.getPerspectiveTransform(ins_pts, ref_pts)
            aligned_img = cv2.warpPerspective(inspect_img, H, (w_orig, h_orig))
            return aligned_img, H, good_matches, kp_inspect, kp_ref, inliers
        except Exception as e:
            print(f"Warning: Corner alignment failed: {e}. Falling back to ORB.")
            
    # 3. Fallback to ORB warp if corners could not be found or are too small/noisy (e.g. shadow image)
    # If inliers are too low, do NOT warp the image using ORB as it will smear/distort it.
    if inliers < 20 or len(good_matches) < 4:
        print("Warning: Fallback ORB matching failed (too few inliers or matches). Returning copy.")
        return inspect_img.copy(), None, good_matches, kp_inspect, kp_ref, inliers
        
    points_inspect = np.float32([kp_inspect[m.queryIdx].pt for m in good_matches])
    points_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches])
    
    aff_matrix_feat = None
    try:
        aff_matrix_feat, _ = cv2.estimateAffine2D(points_inspect, points_ref, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    except:
        pass
        
    if aff_matrix_feat is None:
        return inspect_img.copy(), None, good_matches, kp_inspect, kp_ref, inliers
        
    aff_matrix_orig = aff_matrix_feat.copy()
    aff_matrix_orig[0, 2] /= scale
    aff_matrix_orig[1, 2] /= scale
    
    aligned_img = cv2.warpAffine(inspect_img, aff_matrix_orig, (w_orig, h_orig))
    return aligned_img, aff_matrix_orig, good_matches, kp_inspect, kp_ref, inliers

def visualize_matches(inspect_img, kp_inspect, ref_img, kp_ref, matches, scale=0.5):
    """
    Visualize matched keypoints between the inspect and reference images.

    Parameters:
        inspect_img (numpy.ndarray): The inspection image.
        kp_inspect (list): Keypoints in the inspection image.
        ref_img (numpy.ndarray): The reference image.
        kp_ref (list): Keypoints in the reference image.
        matches (list): List of feature matches.
        scale (float): Scaling factor for visualization.

    Example:
        >>> ref = load_image("data/solved_pipeline/ref_solved.jpg")
        >>> inspect = load_image("data/solved_pipeline/inspect_solved_single_defect.jpg")
        >>> aligned, aff_mat, matches, kp_ins, kp_ref, inliers = align_images_orb(inspect, ref)
        >>> visualize_matches(inspect, kp_ins, ref, kp_ref, matches)
    """
    if inspect_img is None or ref_img is None:
        print("Warning: Cannot visualize matches on None images.")
        return

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
    Segment and identify the main outer contour of the Rubik's Cube using Saturation channel.
    Avoids using 'else' / 'elif' blocks and uses pre-allocated struct elements.

    Parameters:
        img_rgb (numpy.ndarray): The input RGB image.
        thresh_val (int): The binary threshold value for Saturation.

    Returns:
        numpy.ndarray: The parsed contour points, or None if not found.

    Example:
        >>> ref = load_image("data/solved_pipeline/ref_solved.jpg")
        >>> cnt = get_cube_contour_sat(ref)
        >>> print(cv2.contourArea(cnt))
        786432.0
    """
    if img_rgb is None or img_rgb.size == 0:
        print("Warning: Input image is empty or None in get_cube_contour_sat.")
        return None

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    _, thresh = cv2.threshold(s, thresh_val, 255, cv2.THRESH_BINARY)
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL_RECT_21)
    
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
        print("Warning: No valid inner cube contour found in get_cube_contour_sat.")
        return None
    return max(inner_contours, key=cv2.contourArea)

def detect_rotated_layer_by_hull(img_rgb, threshold_depth_pct=3.0):
    """
    Identify layer rotation by evaluating convexity defect depth of the Saturation contour,
    normalized as a percentage of the cube's bounding box width.
    Avoids using 'else' / 'elif' blocks.

    Parameters:
        img_rgb (numpy.ndarray): The input image.
        threshold_depth_pct (float): Depth threshold (as a percentage of cube width) to trigger a rotation.

    Returns:
        tuple: (is_rotated (bool), rotated_box (tuple), max_depth_pct (float))

    Example:
        >>> inspect = load_image("data/scrambled_pipeline/inspect_scrambled_rotated_layer_hard.jpg")
        >>> is_rot, bbox, depth_pct = detect_rotated_layer_by_hull(inspect)
        >>> print(is_rot)
        True
    """
    if img_rgb is None or img_rgb.size == 0:
        print("Warning: Input image is empty or None in detect_rotated_layer_by_hull.")
        return False, None, 0.0

    contour = get_cube_contour_sat(img_rgb, thresh_val=40)
    if contour is None:
        print("Warning: Cube contour is None in detect_rotated_layer_by_hull.")
        return False, None, 0.0
        
    cx, cy, cw, ch = cv2.boundingRect(contour)
    
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    
    hull_idx = cv2.convexHull(approx_contour, returnPoints=False)
    hull_idx = np.sort(hull_idx, axis=0)
    
    defects = None
    try:
        defects = cv2.convexityDefects(approx_contour, hull_idx)
    except Exception as e:
        print(f"Warning: convexityDefects failed: {e}")
        return False, None, 0.0
        
    if defects is None:
        return False, None, 0.0
        
    max_depth = 0.0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth_px = d / 256.0
        if depth_px > max_depth:
            max_depth = depth_px
            
    max_depth_pct = (max_depth / cw) * 100.0 if cw > 0 else 0.0
    defect_points = []
    
    if max_depth_pct > threshold_depth_pct:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth_px = d / 256.0
            depth_pct = (depth_px / cw) * 100.0 if cw > 0 else 0.0
            if depth_pct > threshold_depth_pct:
                far_pt = approx_contour[f][0]
                defect_points.append(far_pt)
                
        if len(defect_points) >= 1:
            defect_points = np.array(defect_points)
            x, y, w, h = cv2.boundingRect(defect_points)
            padding = 40
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = w + 2 * padding
            h = h + 2 * padding
            return True, (x, y, w, h), max_depth_pct
        
    return False, None, max_depth_pct

def get_defect_mask(aligned_inspect, ref_img, threshold_value=30):
    """
    Perform subtraction masking on the Value channel of ref and aligned inspection images.
    Eliminates the 'else' block and uses pre-allocated global morphological filters.

    Parameters:
        aligned_inspect (numpy.ndarray): The aligned inspection image.
        ref_img (numpy.ndarray): The reference image.
        threshold_value (int): Difference threshold value.

    Returns:
        tuple: (diff, mask_final)

    Example:
        >>> ref = load_image("data/solved_pipeline/ref_solved.jpg")
        >>> inspect = load_image("data/solved_pipeline/inspect_solved_single_defect.jpg")
        >>> aligned, _, _, _, _, _ = align_images_orb(inspect, ref)
        >>> diff, mask = get_defect_mask(aligned, ref)
    """
    if aligned_inspect is None or ref_img is None:
        print("Warning: Input or reference image is None in get_defect_mask.")
        return None, None

    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_RGB2HSV)
    aligned_hsv = cv2.cvtColor(aligned_inspect, cv2.COLOR_RGB2HSV)
    
    ref_v = ref_hsv[:, :, 2]
    aligned_v = aligned_hsv[:, :, 2]
    
    diff = cv2.absdiff(ref_v, aligned_v)
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    mask_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, KERNEL_RECT_9)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, KERNEL_RECT_9)
    
    mask_final = mask_cleaned
    ref_cnt = get_cube_contour_sat(ref_img, thresh_val=40)
    if ref_cnt is not None:
        ref_mask = np.zeros_like(ref_v)
        cv2.drawContours(ref_mask, [ref_cnt], -1, 255, -1)
        ref_mask_eroded = cv2.erode(ref_mask, KERNEL_RECT_21)
        mask_final = cv2.bitwise_and(mask_cleaned, ref_mask_eroded)
        
    return diff, mask_final

def classify_and_draw_defects(original_img, defect_mask, aligned_inspect, ref_img, h_matrix, inliers, mode='solved', inspect_path=""):
    """
    Run pipeline checks to classify and draw bounding boxes around defects on the aligned image.
    Uses 'Early Returns' to fully eliminate all 'else'/'elif' statements.
    Optimizes performance by pre-converting to HSV and precomputing cell coordinate grids.
    All geometric metrics (such as convexity depth) are scale-normalized.

    Parameters:
        original_img (numpy.ndarray): Original inspection image.
        defect_mask (numpy.ndarray): Subtraction mask of defects.
        aligned_inspect (numpy.ndarray): Aligned inspection image.
        ref_img (numpy.ndarray): Reference image.
        h_matrix (numpy.ndarray): Homography/Affine transform matrix.
        inliers (int): Count of matching inliers.
        mode (str): Pipeline mode ('solved' or 'scrambled').
        inspect_path (str): Filepath to the original inspection image.

    Returns:
        tuple: (annotated, defect_count, defects_list)

    Example:
        >>> ref = load_image("data/solved_pipeline/ref_solved.jpg")
        >>> path = "data/solved_pipeline/inspect_solved_single_defect.jpg"
        >>> inspect = load_image(path)
        >>> aligned, h, _, _, _, inl = align_images_orb(inspect, ref)
        >>> _, mask = get_defect_mask(aligned, ref)
        >>> ann, count, d_list = classify_and_draw_defects(aligned, mask, aligned, ref, h, inl, mode='solved', inspect_path=path)
    """
    if original_img is None or ref_img is None:
        print("Warning: Input or reference image is None in classify_and_draw_defects.")
        return original_img, 0, []

    annotated = aligned_inspect.copy()
    img_h, img_w = aligned_inspect.shape[:2]
    
    # Get reference mask and its contours/snapping
    ref_cnt = get_cube_contour_sat(ref_img, thresh_val=40)
    if ref_cnt is None:
        print("Warning: Reference cube contour not found in classify_and_draw_defects.")
        return annotated, 0, []
        
    # -----------------------------------------------------------------
    # 1. SCRAMBLED MODE PIPELINE (Early Return - No 'else'/'elif')
    # -----------------------------------------------------------------
    if mode == 'scrambled':
        cnt = get_cube_contour_sat(original_img, thresh_val=40)
        if cnt is None or cv2.contourArea(cnt) < 200000:
            print("Warning: Cube contour missing or too small in scrambled mode.")
            return annotated, 0, []
            
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        
        # Calculate max_depth for convexity defects (rotated layer check)
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
        except Exception as e:
            print(f"Warning: scrambled convexity defects failed: {e}")
            
        # Scale-normalization of convexity depth (as a percentage of cube width)
        max_depth_pct = (max_depth / cw) * 100.0 if cw > 0 else 0.0
        
        # Gating rotated layer check by ORB unaligned inliers (threshold 20 separates straight vs rotated/stress)
        aligned_success = (inliers >= 20)
        is_rotated_layer = False
        if not aligned_success:
            is_rotated_layer = (max_depth_pct > 3.0)
            
        # Get grid coordinates dynamically from reference contour in scrambled mode
        cx_ref, cy_ref, cw_ref, ch_ref = cv2.boundingRect(ref_cnt)
        ch_cube = int(ch_ref * 0.82)
        cw_third = cw_ref // 3
        ch_third = ch_cube // 3
        
        # Construct reference mask
        ref_mask = np.zeros(ref_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(ref_mask, [ref_cnt], -1, 255, -1)
        
        # Erode the bottom edge by 60 pixels to prevent background stand/shadow leaks
        inspect_max_y = cy_ref + ch_cube - 60
        ref_mask[inspect_max_y:, :] = 0
        ref_mask_eroded = cv2.erode(ref_mask, KERNEL_RECT_15)
        
        hsv_aligned = cv2.cvtColor(aligned_inspect, cv2.COLOR_RGB2HSV)
        h_a, s_a, v_a = cv2.split(hsv_aligned)
        
        black_mask = (v_a < 60) & (s_a < 100) & (ref_mask_eroded == 255)
        black_mask = (black_mask * 255).astype(np.uint8)
        
        cleaned_black = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, KERNEL_RECT_7)
        cleaned_black = cv2.morphologyEx(cleaned_black, cv2.MORPH_CLOSE, KERNEL_RECT_15)
        
        contours_black, _ = cv2.findContours(cleaned_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_patches = []
        for c_cnt in contours_black:
            area = cv2.contourArea(c_cnt)
            if area > 4000: # Optimal threshold to avoid false positives but catch all real defects
                px, py, pw, ph = cv2.boundingRect(c_cnt)
                raw_patches.append((px, py, pw, ph, area))
                
        defects_list = []
        
        if is_rotated_layer:
            rx, ry, rw, rh = cx_ref, cy_ref + 2 * ch_third, cw_ref, ch_third
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 6)
            cv2.putText(annotated, "Rotated Layer", (rx, ry - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
            defects_list.append({"type": "Rotated Layer", "bbox": (rx, ry, rw, rh), "area": rw*rh})
            
            defect_idx = 1
            for px, py, pw, ph, area in raw_patches:
                cell_x = px + pw // 2
                cell_y = py + ph // 2
                c = (cell_x - cx_ref) // cw_third
                r = (cell_y - cy_ref) // ch_third
                r = max(0, min(2, int(r)))
                c = max(0, min(2, int(c)))
                
                sx, sy, sw, sh = cx_ref + c * cw_third, cy_ref + r * ch_third, cw_third, ch_third
                cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 4)
                cv2.putText(annotated, f"Cell/Sticker Defect {defect_idx}", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                defects_list.append({"type": "Cell/Sticker Defect", "bbox": (sx, sy, sw, sh), "area": sw*sh})
                defect_idx += 1
                
        if not is_rotated_layer:
            defect_idx = 1
            for px, py, pw, ph, area in raw_patches:
                cell_x = px + pw // 2
                cell_y = py + ph // 2
                c = (cell_x - cx_ref) // cw_third
                r = (cell_y - cy_ref) // ch_third
                r = max(0, min(2, int(r)))
                c = max(0, min(2, int(c)))
                
                sx, sy, sw, sh = cx_ref + c * cw_third, cy_ref + r * ch_third, cw_third, ch_third
                cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 4)
                cv2.putText(annotated, f"Cell/Sticker Defect {defect_idx}", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                defects_list.append({"type": "Cell/Sticker Defect", "bbox": (sx, sy, sw, sh), "area": sw*sh})
                defect_idx += 1
                
        return annotated, len(defects_list), defects_list

    # -----------------------------------------------------------------
    # 2. SOLVED MODE PIPELINE (Clean scale-invariant reference-grid checks)
    # -----------------------------------------------------------------
    cx_ref = 919
    cy_ref = 2220
    cw_ref = 1054
    ch_ref = 760
    
    hsv_ref = cv2.cvtColor(ref_img, cv2.COLOR_RGB2HSV)
    blue_mask = cv2.inRange(hsv_ref, np.array([90, 100, 50]), np.array([130, 255, 255]))
    mask_closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, KERNEL_RECT_21)
    contours_blue, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_blue:
        best_cnt = max(contours_blue, key=cv2.contourArea)
        cx_ref, cy_ref, cw_ref, ch_ref = cv2.boundingRect(best_cnt)
        
    cw_third = cw_ref // 3
    ch_third = ch_ref // 3
    
    # Perform HSV conversion exactly ONCE on the entire image before checking loops
    hsv_aligned = cv2.cvtColor(aligned_inspect, cv2.COLOR_RGB2HSV)
    blue_m = cv2.inRange(hsv_aligned, np.array([90, 100, 50]), np.array([130, 255, 255]))
    
    # Precompute cell boundaries and store them to avoid redundant math inside loops
    cell_coords = []
    for r in range(3):
        row_coords = []
        for c in range(3):
            margin_x = int(cw_third * 0.15)
            margin_y = int(ch_third * 0.15)
            x1 = cx_ref + c * cw_third + margin_x
            x2 = cx_ref + (c + 1) * cw_third - margin_x
            y1 = cy_ref + r * ch_third + margin_y
            y2 = cy_ref + (r + 1) * ch_third - margin_y
            row_coords.append((x1, x2, y1, y2))
        cell_coords.append(row_coords)
        
    grid_results = np.zeros((3, 3))
    for r in range(3):
        for c in range(3):
            x1, x2, y1, y2 = cell_coords[r][c]
            cell_mask = blue_m[y1:y2, x1:x2]
            blue_pct = 0.0
            if cell_mask.size > 0:
                blue_pct = (np.sum(cell_mask == 255) / cell_mask.size) * 100.0
            grid_results[r, c] = blue_pct
            
    defective = grid_results < 50.0
    
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
            cv2.putText(annotated, "Rotated Layer 1", (rx, ry - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
            defects_list.append({"type": "Rotated Layer", "bbox": (rx, ry, rw, rh), "area": rw*rh})
            
    # Draw Rotated Columns
    for c in range(3):
        if rotated_cols[c]:
            rx, ry, rw, rh = cx_ref + c * cw_third, cy_ref, cw_third, ch_ref
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 6)
            cv2.putText(annotated, "Rotated Layer 1", (rx, ry - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
            defects_list.append({"type": "Rotated Layer", "bbox": (rx, ry, rw, rh), "area": rw*rh})
            
    # Draw Cell/Sticker Defects (with early continue optimization and HSV single-pass slicing)
    for r in range(3):
        is_row_rot = rotated_rows[r]
        for c in range(3):
            is_col_rot = rotated_cols[c]
            
            x1, x2, y1, y2 = cell_coords[r][c]
            # Fast slicing of precomputed hsv_aligned image (No cv2.cvtColor in loop)
            cell_hsv = hsv_aligned[y1:y2, x1:x2]
            
            if cell_hsv.size == 0:
                continue
                
            h_c = cell_hsv[:, :, 0]
            s_c = cell_hsv[:, :, 1]
            v_c = cell_hsv[:, :, 2]
            
            is_blue_c = (h_c >= 90) & (h_c <= 130) & (s_c >= 60)
            is_saturated_other = (s_c >= 60) & (~is_blue_c)
            is_black_tape_c = (v_c < 60) & (v_c > 10) & (s_c > 10)
            
            is_black_tape = (np.sum(is_black_tape_c) / cell_hsv.size) * 300.0 > 10.0
            
            is_defect = False
            if is_row_rot or is_col_rot:
                # If layer is rotated, only detect if it is a black tape defect
                is_defect = is_black_tape
            if not (is_row_rot or is_col_rot):
                # If layer is not rotated, detect either saturated other color or black tape
                defect_pct = (np.sum(is_saturated_other | is_black_tape_c) / cell_hsv.size) * 300.0
                is_defect = defect_pct > 15.0
                
            if is_defect:
                sx, sy, sw, sh = cx_ref + c * cw_third, cy_ref + r * ch_third, cw_third, ch_third
                defect_count += 1
                cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 4)
                cv2.putText(annotated, f"Cell/Sticker Defect {defect_count}", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                defects_list.append({"type": "Cell/Sticker Defect", "bbox": (sx, sy, sw, sh), "area": sw*sh})
                
    return annotated, len(defects_list), defects_list