import json
from tkinter import ttk, messagebox
import tkinter as tk

from helper.language import language


class ModelProperties:
    def __init__(self, properties_panel):
        self.properties_panel = properties_panel
        self.create_combobox_style()

    def create_combobox_style(self):
        style = ttk.Style()
        style.configure("CustomCombobox.TCombobox",
                        borderwidth=1,
                        relief="solid",
                        background="white",
                        selectbackground="white",
                        selectforeground="black")
        style.map("CustomCombobox.TCombobox",
                  fieldbackground=[('readonly', 'white')],
                  background=[('readonly', 'white')],
                  foreground=[('readonly', 'black')])

    def add_property(self, label_text, row, value, is_combobox=False, values=None):
        label = tk.Label(self.properties_panel.properties_frame, text=label_text, bg='white', font=("Helvetica", 12))
        label.grid(row=row, column=0, sticky='w', padx=10, pady=5)

        if is_combobox:
            combobox = ttk.Combobox(self.properties_panel.properties_frame, values=values, font=("Helvetica", 12),
                                    width=10, style="CustomCombobox.TCombobox")  # Adjusted width to 10
            combobox.set(value)
            combobox.grid(row=row, column=1, padx=10, pady=5)
            setattr(self, label_text, combobox)
        else:
            entry = tk.Entry(self.properties_panel.properties_frame, font=("Helvetica", 12), width=11,
                             highlightthickness=1, highlightbackground='black',
                             highlightcolor='black')  # Adjusted width to 11
            entry.insert(0, value)
            entry.grid(row=row, column=1, padx=10, pady=5)
            setattr(self, label_text, entry)

    def show_model_properties(self, model_info):
        self.properties_panel.show_empty_properties()

        # Display model name at the top
        model_name_label = tk.Label(self.properties_panel.properties_frame,
                                    text=language.translate("Model name") + model_info['name'],
                                    bg='white', font=("Helvetica", 14, 'bold'))
        model_name_label.grid(row=0, column=0, columnspan=2, pady=(10, 10))

        self.add_property(language.translate("width"), 1, model_info.get('width', ''))
        self.add_property(language.translate("height"), 2, model_info.get('height', ''))
        self.add_property(language.translate("centerX"), 3, model_info.get('centerX', ''))
        self.add_property(language.translate("centerY"), 4, model_info.get('centerY', ''))
        self.add_property(language.translate("radius"), 5, model_info.get('radius', ''))
        self.add_property(language.translate("diameter"), 6, model_info.get('diameter', ''))
        self.add_property(language.translate("rotation_angle"), 7, model_info.get('rotation_angle', ''))
        self.add_property(language.translate("matching"), 8, model_info.get('matching', ''))
        self.add_property(language.translate("detection_order"), 9, model_info.get('detection_order', ''),
                          is_combobox=True,
                          values=["Ascending X", "Descending X", "Ascending Y", "Descending Y", "Maximum Matching %"])
        self.add_property(language.translate("detection_count"), 10, model_info.get('Count', ''))

        # Add update button
        update_button = tk.Button(self.properties_panel.properties_frame, text=language.translate("Update"),
                                  command=lambda: self.update_model_properties(model_info['name']),
                                  font=("Helvetica", 12))
        update_button.grid(row=11, column=0, columnspan=2, pady=(10, 10))

    def update_model_properties(self, model_name):
        updated_properties = {
            'width': self.width.get(),
            'height': self.height.get(),
            'centerX': self.centerX.get(),
            'centerY': self.centerY.get(),
            'radius': self.radius.get(),
            'diameter': self.diameter.get(),
            'rotation_angle': self.rotation_angle.get(),
            'matching': self.matching.get(),
            'detection_order': self.detection_order.get(),
            'Count': self.detection_count.get()
        }

        json_path = "model_info.json"
        try:
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        for model in data:
            if model.get('name') == model_name:
                model.update(updated_properties)
                break

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        messagebox.showinfo("Update", "Model properties updated successfully.")
