import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageOps
import math
import json


def extract_objects(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    edged = cv2.Canny(thresh, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        obj = image[y:y + h, x:x + w]

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        objects.append((obj, (x, y, w, h), (cx, cy), cnt))

    return objects


def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    # If the boxes are in integer form, convert them to float for calculations
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]  # Index of the box with the largest bottom-right y-coordinate
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the box and the smallest (x, y) coordinates for the
        # end of the box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the overlapping area
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indices where the overlap is greater than the threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick].astype("int")


def sort_objects_by_order(centers, match_percentages, contours, order):
    if order == "Ascending X":
        sorted_objects = sorted(zip(centers, match_percentages, contours), key=lambda x: x[0][0])
    elif order == "Descending X":
        sorted_objects = sorted(zip(centers, match_percentages, contours), key=lambda x: x[0][0], reverse=True)
    elif order == "Ascending Y":
        sorted_objects = sorted(zip(centers, match_percentages, contours), key=lambda x: x[0][1])
    elif order == "Descending Y":
        sorted_objects = sorted(zip(centers, match_percentages, contours), key=lambda x: x[0][1], reverse=True)
    elif order == "Maximum Matching %":
        sorted_objects = sorted(zip(centers, match_percentages, contours), key=lambda x: x[1], reverse=True)
    else:
        sorted_objects = list(zip(centers, match_percentages, contours))  # Default order if no match

    return zip(*sorted_objects)


def find_and_match_object(reference_image_path, larger_image_path, threshold=0.8, overlap_thresh=0.3):
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    larger_image = cv2.imread(larger_image_path, cv2.IMREAD_GRAYSCALE)

    if reference_image is None or larger_image is None:
        print("Error loading images.")
        return [], [], [], 0

    reference_objects = extract_objects(reference_image)
    larger_objects = extract_objects(larger_image)

    print(f"Extracted {len(reference_objects)} objects from the reference image.")
    print(f"Extracted {len(larger_objects)} objects from the larger image.")

    boxes = []
    scores = []
    centers = []
    contours = []
    count_10_percent = 0

    for (larger_obj, (lx, ly, lw, lh), (lcx, lcy), larger_cnt) in larger_objects:
        best_score = 0
        best_box = None
        best_contour = None

        for (ref_obj, (rx, ry, rw, rh), (rcx, rcy), ref_cnt) in reference_objects:
            if ref_obj.size == 0 or larger_obj.size == 0:
                continue

            resized_larger_obj = cv2.resize(larger_obj, (ref_obj.shape[1], ref_obj.shape[0]))

            win_size = min(resized_larger_obj.shape[0], resized_larger_obj.shape[1], 7)
            if win_size < 7:
                continue

            ssim_index = ssim(ref_obj, resized_larger_obj, win_size=win_size)

            if ssim_index > best_score:
                best_score = ssim_index
                best_box = (lx, ly, lx + lw, ly + lh)
                center = (lcx, lcy)
                best_contour = larger_cnt

        if best_score * 100 >= threshold * 100 and best_box is not None:
            boxes.append(best_box)
            scores.append(best_score * 100)
            centers.append(center)
            contours.append(best_contour)

        if best_score * 100 >= 10:
            count_10_percent += 1

    if len(boxes) > 0:
        boxes = non_max_suppression(np.array(boxes), overlap_thresh)

    print(f"Found {len(boxes)} matching objects.")
    print(f"Objects matching >= 10%: {count_10_percent}")

    return boxes, scores, centers, contours, count_10_percent


def adjust_box_position(px, py, cx, cy, half_box_size):
    vx = px - cx
    vy = py - cy
    mag = np.sqrt(vx ** 2 + vy ** 2)
    vx /= mag
    vy /= mag
    px = int(px + vx * half_box_size)
    py = int(py + vy * half_box_size)
    ox = int(cx + (cx - px))
    oy = int(cy + (cy - py))
    return (px, py, ox, oy)


def move_box_to_best_position(px, py, cx, cy, half_box_size, image_shape, image_array, other_centers):
    angle = 0
    max_angle = 360
    step = 20  # 10
    while angle < max_angle:
        angle_rad = np.radians(angle)
        new_px = int(cx + np.cos(angle_rad) * (px - cx) - np.sin(angle_rad) * (py - cy))
        new_py = int(cy + np.sin(angle_rad) * (px - cx) + np.cos(angle_rad) * (py - cy))

        new_px = np.clip(new_px, half_box_size, image_shape[1] - half_box_size)
        new_py = np.clip(new_py, half_box_size, image_shape[0] - half_box_size)

        ox = int(cx + (cx - new_px))
        oy = int(cy + (cy - new_py))

        ox = np.clip(ox, half_box_size, image_shape[1] - half_box_size)
        oy = np.clip(oy, half_box_size, image_shape[0] - half_box_size)

        # Check proximity to other centers and white pixels
        if not is_box_overlapping_with_others(new_px, new_py, ox, oy, half_box_size, other_centers, image_array):
            return new_px, new_py, ox, oy

        angle += step

    # If no good position is found, return original position
    return px, py, cx + (cx - px), cy + (cy - py)


def has_white_under_box(image_array, px, py, half_box_size):
    box = image_array[py - half_box_size:py + half_box_size, px - half_box_size:px + half_box_size]
    return np.any(box == 255)


def calculate_and_display_matches(image_view, reference_image_path, larger_image_path, model_name):
    binary_reference_image_path = convert_to_binary(reference_image_path)
    binary_larger_image_path = convert_to_binary(larger_image_path)

    # Load the model JSON file to get the box_size and detection_order
    json_path = "model_info.json"
    default_box_size = 50  # Set your default box size here
    detection_order = "Ascending X"  # Default detection order
    box_size = default_box_size

    try:
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
            # Find the model by name and get its box_size and detection_order if present
            for model in data:
                if model.get('name') == model_name:
                    box_size = model.get('box_size', default_box_size)
                    detection_order = model.get('detection_order', detection_order)
                    break
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error loading model info. Using default values.")

    boxes, match_percentages, centers, contours, total_10_percent_objects = find_and_match_object(
        binary_reference_image_path, binary_larger_image_path, threshold=0.8, overlap_thresh=0.3
    )
    print(f"Match Percentages: {match_percentages}")

    if not match_percentages:
        print("No matches found.")
        return

    # Sort objects based on detection order
    centers, match_percentages, contours = sort_objects_by_order(centers, match_percentages, contours, detection_order)

    detected_image = cv2.imread(larger_image_path)
    detected_image_pil = Image.fromarray(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(detected_image_pil)

    for i, (center, score, contour) in enumerate(zip(centers, match_percentages, contours)):
        if score >= 35:
            radius = 5
            center_x, center_y = center

            # Draw object index
            draw.text((center_x - 15, center_y - 15), f'#{i + 1}', fill="blue")

            # Draw match percentage and center point
            draw.text((center_x + 15, center_y - 15), f'{score:.2f}%', fill="green")
            draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill="red")

    draw_detected_object_boxes(draw, centers, contours, box_size,
                               cv2.imread(binary_larger_image_path, cv2.IMREAD_GRAYSCALE), centers)

    draw.text((10, 30), f'Total Objects Matching >= 10%: {total_10_percent_objects}', fill="blue")

    image_view.add_thumbnail(detected_image_pil)
    image_view.update_image()


def convert_to_binary(image_path):
    image = Image.open(image_path)
    binary_image = ImageOps.grayscale(image)
    binary_image = binary_image.point(lambda x: 0 if x < 128 else 255, '1')
    binary_image_path = f"images/gray/binary_{os.path.basename(image_path)}"
    binary_image.save(binary_image_path)
    return binary_image_path


def draw_detected_object_boxes(draw, centers, contours, box_size, image_array, other_centers):
    # Dynamically calculate the circle_offset based on box_size
    circle_offset = 0.5 * box_size + 5

    half_box_size = box_size // 2
    for center, contour in zip(centers, contours):
        if len(contour) > 0:
            cx, cy = center
            point = contour[0][0]
            px, py = point

            # Adjust the position so that the connecting line is either horizontal or vertical
            if abs(px - cx) > abs(py - cy):
                py = cy  # Make it a horizontal line
            else:
                px = cx  # Make it a vertical line

            # Adjust box positions
            px, py, ox, oy = adjust_box_position(px, py, cx, cy, half_box_size)
            px, py, ox, oy = move_box_to_best_position(px, py, cx, cy, half_box_size, image_array.shape, image_array,
                                                       other_centers)

            if is_box_overlapping_with_others(px, py, ox, oy, half_box_size, other_centers, image_array):
                continue

            # Calculate box center positions for connecting line
            box1_center_x, box1_center_y = px, py
            box2_center_x, box2_center_y = ox, oy

            # Calculate the angle between each box center and the object center
            box1_angle = calculate_angle(box1_center_x, box1_center_y, cx, cy)
            box2_angle = calculate_angle(box2_center_x, box2_center_y, cx, cy)

            # Find the nearest right angle (0°, 90°, 180°, 270°)
            adjusted_box1_angle = align_to_nearest_angle(box1_angle)
            adjusted_box2_angle = align_to_nearest_angle(box2_angle)

            # Continuously adjust the angle until the difference is zero
            while True:
                angle1_diff = calculate_angle(box1_center_x, box1_center_y, cx, cy) - adjusted_box1_angle
                angle2_diff = calculate_angle(box2_center_x, box2_center_y, cx, cy) - adjusted_box2_angle

                if abs(angle1_diff) > 0:
                    adjusted_box1_angle += angle1_diff

                if abs(angle2_diff) > 0:
                    adjusted_box2_angle += angle2_diff

                # If the difference is very small (close to zero), stop adjusting
                if abs(angle1_diff) < 0.1 and abs(angle2_diff) < 0.1:
                    break

            # Drawing the rotated boxes
            draw_rotated_rectangle(draw, box1_center_x, box1_center_y, box_size, box_size, adjusted_box1_angle, "red")
            draw_rotated_rectangle(draw, box2_center_x, box2_center_y, box_size, box_size, adjusted_box2_angle, "red")

            # Drawing connecting lines from box center to object center
            draw.line([box1_center_x, box1_center_y, cx, cy], fill="red", width=3)
            draw.line([box2_center_x, box2_center_y, cx, cy], fill="red", width=3)

            # Drawing ellipses for points
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill="blue")

            # Drawing circles instead of rectangles around objects with dynamically calculated offset
            radius = int(np.linalg.norm(np.array([px, py]) - np.array([cx, cy])) - circle_offset)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="yellow", width=2)

            # Display the adjusted angles next to each box
            draw.text((box1_center_x + 15, box1_center_y - 15), f'{adjusted_box1_angle:.2f}°', fill="yellow")
            draw.text((box2_center_x + 15, box2_center_y - 15), f'{adjusted_box2_angle:.2f}°', fill="yellow")

            # Display the angle difference between box center line and connecting line
            draw.text((box1_center_x - 15, box1_center_y + 15), f'Diff: {angle1_diff:.2f}°', fill="cyan")
            draw.text((box2_center_x - 15, box2_center_y + 15), f'Diff: {angle2_diff:.2f}°', fill="cyan")


def is_box_overlapping_with_others(px, py, ox, oy, half_box_size, other_centers, image_array):
    """Check if the box overlaps with other objects."""
    for center_x, center_y in other_centers:
        if np.linalg.norm(np.array([px, py]) - np.array([center_x, center_y])) < 2 * half_box_size or \
                np.linalg.norm(np.array([ox, oy]) - np.array([center_x, center_y])) < 2 * half_box_size:
            return True

    # Additionally check if the box overlaps with any white pixels (indicating other objects)
    if has_white_under_box(image_array, px, py, half_box_size) or has_white_under_box(image_array, ox, oy,
                                                                                      half_box_size):
        return True

    return False


def align_to_nearest_angle(angle):
    """
    Adjust the angle to ensure that the red box aligns its center line
    with the connecting line at 0°, 90°, 180°, or 270°.
    """
    possible_angles = [0, 90, 180, 270]
    closest_angle = min(possible_angles, key=lambda x: abs(x - angle))
    return closest_angle


def calculate_angle(px1, py1, px2, py2):
    """Calculate the angle between two points."""
    delta_x = px2 - px1
    delta_y = py2 - py1
    angle = math.degrees(math.atan2(delta_y, delta_x))

    # Convert to positive angle if necessary
    if angle < 0:
        angle += 360

    return angle


def draw_rotated_rectangle(draw, center_x, center_y, width, height, angle, outline_color, outline_width=3):
    """Draw a rotated rectangle around the given center point with a specified outline width."""
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Calculate the four corners of the rectangle
    half_width = width // 2
    half_height = height // 2

    corners = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height)
    ]

    # Calculate the rotated corners
    rotated_corners = []
    for corner in corners:
        x = center_x + corner[0] * cos_a - corner[1] * sin_a
        y = center_y + corner[0] * sin_a + corner[1] * cos_a
        rotated_corners.append((x, y))

    # Draw the rotated rectangle multiple times with a slight offset to simulate thicker borders
    for i in range(outline_width):
        offset = i - (outline_width // 2)
        offset_corners = [(x + offset, y + offset) for (x, y) in rotated_corners]
        draw.polygon(offset_corners, outline=outline_color)
