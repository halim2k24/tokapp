import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageOps


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

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


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


def move_box_away(px, py, cx, cy, half_box_size, image_shape, image_array):
    angle = 0
    max_angle = 360
    step = 10  # Step by 10 degrees

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

        if not has_white_under_box(image_array, new_px, new_py, half_box_size) and not has_white_under_box(image_array,
                                                                                                           ox, oy,
                                                                                                           half_box_size):
            return new_px, new_py, ox, oy

        angle += step

    return px, py, cx + (cx - px), cy + (cy - py)


def ensure_boxes_180_degrees(px, py, cx, cy, half_box_size, image_shape, image_array):
    direction_x = np.sign(px - cx)
    direction_y = np.sign(py - cy)

    for offset in range(-half_box_size * 2, half_box_size * 2 + 1, half_box_size):
        for sign in [-1, 1]:
            new_px = px + sign * offset
            new_py = py + sign * offset
            new_px = np.clip(new_px, half_box_size, image_shape[1] - half_box_size)
            new_py = np.clip(new_py, half_box_size, image_shape[0] - half_box_size)

            ox = cx + (cx - new_px)
            oy = cy + (cy - new_py)
            ox = np.clip(ox, half_box_size, image_shape[1] - half_box_size)
            oy = np.clip(oy, half_box_size, image_shape[0] - half_box_size)

            if not has_white_under_box(image_array, new_px, new_py, half_box_size) and not has_white_under_box(
                    image_array, ox, oy, half_box_size):
                return new_px, new_py, ox, oy

    return px, py, cx, cy


def draw_detected_object_boxes(draw, centers, contours, box_size, image_array):
    half_box_size = box_size // 2
    for center, contour in zip(centers, contours):
        if len(contour) > 0:
            cx, cy = center
            point = contour[0][0]
            px, py = point

            px, py, ox, oy = adjust_box_position(px, py, cx, cy, half_box_size)

            px, py, ox, oy = move_box_away(px, py, cx, cy, half_box_size, image_array.shape, image_array)

            draw.rectangle([px - half_box_size, py - half_box_size, px + half_box_size, py + half_box_size],
                           outline="red", width=2)
            draw.rectangle([ox - half_box_size, oy - half_box_size, ox + half_box_size, oy + half_box_size],
                           outline="red", width=2)
            draw.line([px, py, cx, cy], fill="red", width=1)
            draw.line([ox, oy, cx, cy], fill="red", width=1)
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="green")
            draw.ellipse((ox - 2, oy - 2, ox + 2, oy + 2), fill="green")
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill="blue")


def has_white_under_box(image_array, px, py, half_box_size):
    box = image_array[py - half_box_size:py + half_box_size, px - half_box_size:px + half_box_size]
    return np.any(box == 255)


def calculate_and_display_matches(image_view, reference_image_path, larger_image_path):
    binary_reference_image_path = convert_to_binary(reference_image_path)
    binary_larger_image_path = convert_to_binary(larger_image_path)

    boxes, match_percentages, centers, contours, total_10_percent_objects = find_and_match_object(
        binary_reference_image_path, binary_larger_image_path, threshold=0.8, overlap_thresh=0.3
    )
    print(f"Match Percentages: {match_percentages}")

    if not match_percentages:
        print("No matches found.")
        return

    detected_image = cv2.imread(larger_image_path)
    detected_image_pil = Image.fromarray(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(detected_image_pil)

    for i, (box, score, center) in enumerate(zip(boxes, match_percentages, centers)):
        if score >= 35:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="green", width=2)
            draw.text((box[0], box[1] - 10), f'{score:.2f}%', fill="green")

            radius = 5
            center_x, center_y = center
            draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill="red")
            draw.text((center_x + radius + 5, center_y - 10), f'({center_x}, {center_y})', fill="red")
            draw.text((box[0], box[1] - 20), f'#{i + 1}', fill="blue")

    box_size = 50  # Adjust the box size as needed
    draw_detected_object_boxes(draw, centers, contours, box_size,
                               cv2.imread(binary_larger_image_path, cv2.IMREAD_GRAYSCALE))

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
