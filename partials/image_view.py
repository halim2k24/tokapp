import math
import os
import tkinter as tk
from datetime import time

from PIL import Image, ImageTk
import json
from helper.language import language
from helper.rotating_rectangle import RotatingRectangle
import time  # Ensure time module is imported
import cv2  # Add this import
import numpy as np  # Add this import


class ImageView(tk.Frame):
    def __init__(self, parent, shared_image):
        super().__init__(parent, bg='white', relief='solid', borderwidth=1)
        self.place(relwidth=0.6, relheight=1.0, relx=0.21, rely=0.0)

        self.shared_image = shared_image
        self.image_id = None
        self.zoom_factor = 1.0
        self.current_image = None
        self.thumbnails = []
        self.selected_thumbnail = None
        self.rotating_rectangle = None

        self.create_widgets()

    def create_widgets(self):
        self.title_label_b = tk.Label(self, text=language.translate("image_view"), bg='white', font=("Helvetica", 16))
        self.title_label_b.pack(pady=10)

        self.line_b = tk.Frame(self, bg="black", height=2)
        self.line_b.pack(fill='x', pady=(5, 0))

        self.canvas = tk.Canvas(self, bg='white')
        self.canvas.pack(fill='both', expand=True)

        self.thumbnail_frame = tk.Frame(self, bg='white')
        self.thumbnail_frame.pack(fill='x', side='bottom')

        self.scrollbar = tk.Scrollbar(self.thumbnail_frame, orient=tk.HORIZONTAL)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.thumbnail_canvas = tk.Canvas(self.thumbnail_frame, bg='white', height=100,
                                          xscrollcommand=self.scrollbar.set)
        self.thumbnail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.thumbnail_canvas.xview)

        self.thumb_container = tk.Frame(self.thumbnail_canvas, bg='white')
        self.thumb_container.bind("<Configure>", lambda e: self.thumbnail_canvas.configure(
            scrollregion=self.thumbnail_canvas.bbox("all")))
        self.thumbnail_canvas.create_window((0, 0), window=self.thumb_container, anchor="nw")

        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def center_image_on_canvas(self, img_tk):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width = img_tk.width()
        img_height = img_tk.height()
        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2
        self.shared_image.set_position((x, y))
        if self.image_id:
            self.canvas.coords(self.image_id, x, y)
        else:
            self.image_id = self.canvas.create_image(x, y, anchor='nw', image=img_tk)

    def clear_thumbnails(self):
        for widget in self.thumb_container.winfo_children():
            widget.destroy()
        self.thumbnails = []  # Clear the thumbnails list

    def clear_canvas(self):
        for widget in self.thumb_container.winfo_children():
            widget.destroy()
        self.thumbnails = []  # Clear the thumbnails list

    def on_mouse_click(self, event):
        if self.rotating_rectangle and (self.rotating_rectangle.is_point_inside_rectangle(event.x, event.y) or
                                        self.rotating_rectangle.is_point_on_handle(event.x, event.y) or
                                        self.rotating_rectangle.is_point_on_circle(event.x, event.y)):
            self.rotating_rectangle.start_drag(event)
        else:
            self.set_anchor(event)

    def on_mouse_drag(self, event):
        if self.rotating_rectangle and (self.rotating_rectangle.dragging_rectangle or
                                        self.rotating_rectangle.dragging_handle or
                                        self.rotating_rectangle.dragging_circle):
            self.rotating_rectangle.on_drag(event)
        else:
            self.move_image(event)

    def on_mouse_release(self, event):
        if self.rotating_rectangle and (self.rotating_rectangle.dragging_rectangle or
                                        self.rotating_rectangle.dragging_handle or
                                        self.rotating_rectangle.dragging_circle):
            self.rotating_rectangle.stop_drag(event)

    def update_image(self):
        img = self.shared_image.get_image()
        if img:
            pos = self.shared_image.get_position()
            if self.image_id is None:
                self.image_id = self.canvas.create_image(pos[0], pos[1], anchor='nw', image=img)
            else:
                self.canvas.coords(self.image_id, pos)
                self.canvas.itemconfig(self.image_id, image=img)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))


    def add_thumbnail(self, img_full):
        base_width = 100
        w_percent = (base_width / float(img_full.size[0]))
        h_size = int((float(img_full.size[1]) * float(w_percent)))
        thumb_image = img_full.resize((base_width, h_size), Image.Resampling.LANCZOS)
        thumb_image_tk = ImageTk.PhotoImage(thumb_image)
        self.thumbnails.append((thumb_image_tk, img_full))

        thumb_label = tk.Label(self.thumb_container, image=thumb_image_tk, bg='white', borderwidth=2, relief='solid')
        thumb_label.image = thumb_image_tk
        thumb_label.pack(side='left', padx=5, pady=5)
        thumb_label.bind("<Button-1>", lambda e, full_img=img_full: self.on_thumbnail_click(thumb_label, full_img))

    def on_thumbnail_click(self, thumb_label, img_full):
        img_full_tk = ImageTk.PhotoImage(img_full)
        self.shared_image.set_image(img_full_tk)
        self.current_image = img_full
        self.zoom_factor = 1.0
        self.center_image_on_canvas(img_full_tk)
        self.update_image()
        self.highlight_thumbnail(thumb_label)

    def highlight_thumbnail(self, thumb_label):
        if self.selected_thumbnail and self.selected_thumbnail.winfo_exists():
            self.selected_thumbnail.config(bg='white')
        thumb_label.config(bg='green')
        self.selected_thumbnail = thumb_label

    def move_image(self, event):
        x, y = event.x, event.y
        self.canvas.coords(self.image_id, x - self.anchor_x, y - self.anchor_y)
        self.shared_image.set_position((x - self.anchor_x, y - self.anchor_y))

    def set_anchor(self, event):
        bbox = self.canvas.bbox(self.image_id)
        self.anchor_x = event.x - bbox[0]
        self.anchor_y = event.y - bbox[1]

    def enable_rectangle_drawing(self):
        if not self.rotating_rectangle:
            self.rotating_rectangle = RotatingRectangle(self.canvas)
        img_pos = self.shared_image.get_position()
        img_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        image_center = (img_pos[0] + img_size[0] // 2, img_pos[1] + img_size[1] // 2)
        self.rotating_rectangle.center_rectangle_on_image(image_center, self.zoom_factor)

    def update_texts(self):
        self.title_label_b.config(text=language.translate("image_view"))


    def crop_and_save_image(self, model_info):
        if not self.rotating_rectangle or not self.current_image:
            print("Rotating rectangle or current image is missing")
            return

        # Get the bounding box coordinates
        bbox = self.rotating_rectangle.get_rotated_coords()
        print(f"Original image size: {self.current_image.size}")
        print(f"Bounding box for cropping: {bbox}")

        # Calculate the center of the bounding box
        center_x = sum(x for x, y in bbox) / len(bbox)
        center_y = sum(y for x, y in bbox) / len(bbox)

        # Convert canvas coordinates to image coordinates
        canvas_bbox = self.canvas.bbox(self.image_id)
        print(f"Canvas bounding box: {canvas_bbox}")

        # Convert the coordinates relative to the canvas
        bbox = [(x - canvas_bbox[0], y - canvas_bbox[1]) for x, y in bbox]

        # Rotate the image to align the bounding box with axes
        img_array = np.array(self.current_image)
        center = (center_x, center_y)
        angle_rad = math.radians(self.rotating_rectangle.angle)
        rot_mat = cv2.getRotationMatrix2D(center, self.rotating_rectangle.angle, 1.0)
        rotated_img_array = cv2.warpAffine(img_array, rot_mat, (img_array.shape[1], img_array.shape[0]))

        # Calculate the bounding box in the rotated image coordinates
        bbox_rotated = [self.rotate_point((x, y), center, angle_rad) for x, y in bbox]
        x_coords = [x for x, y in bbox_rotated]
        y_coords = [y for x, y in bbox_rotated]
        min_x, max_x = max(0, int(min(x_coords))), min(rotated_img_array.shape[1], int(max(x_coords)))
        min_y, max_y = max(0, int(min(y_coords))), min(rotated_img_array.shape[0], int(max(y_coords)))

        # Crop the rotated image
        cropped_img_array = rotated_img_array[min_y:max_y, min_x:max_x]
        cropped_image = Image.fromarray(cropped_img_array)

        if not os.path.exists("images"):
            os.makedirs("images")

        timestamp = int(time.time())
        cropped_image_path = f"images/{model_info['name']}_{timestamp}.png"
        cropped_image.save(cropped_image_path)

        # Save model info to JSON
        model_info['image_path'] = cropped_image_path
        json_path = "model_info.json"
        self.update_model_info_json(model_info, json_path)

        print(f"Model info saved to {json_path}")
        print(f"Cropped image saved to {cropped_image_path}")

        # Clear the rotating rectangle after saving
        self.rotating_rectangle.clear()

    def rotate_point(self, point, center, angle_rad):
        """ Rotate a point around a given center. """
        x, y = point
        cx, cy = center
        cos_val, sin_val = math.cos(angle_rad), math.sin(angle_rad)
        nx = cos_val * (x - cx) - sin_val * (y - cy) + cx
        ny = sin_val * (x - cx) + cos_val * (y - cy) + cy
        return nx, ny

    def update_model_info_json(self, model_info, json_path):
        try:
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        for model in data:
            if model.get('name') == model_info['name']:
                model.update(model_info)
                break
        else:
            data.append(model_info)

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)