import tkinter as tk
from tkinter import ttk, messagebox
from helper.language import language

class ModelProperties:
    def __init__(self, properties_panel):
        self.properties_panel = properties_panel

    def show_model_properties(self, model_info):
        self.properties_panel.show_empty_properties()

        # Display model name at the top
        model_name_label = tk.Label(self.properties_panel.properties_frame, text="Model name: " + model_info['name'], bg='white', font=("Helvetica", 14, 'bold'))
        model_name_label.grid(row=0, column=0, columnspan=2, pady=(10, 10))

        self.add_property("width", 1, model_info.get('width', ''))
        self.add_property("height", 2, model_info.get('height', ''))
        self.add_property("center_x", 3, model_info.get('center_x', ''))
        self.add_property("center_y", 4, model_info.get('center_y', ''))
        self.add_property("rotation_angle", 5, model_info.get('rotation_angle', ''))
        self.add_property("matching", 6, model_info.get('matching', ''), is_combobox=True, values=["Ascending X", "Descending X", "Ascending Y", "Descending Y", "Maximum Matching %"])
        self.add_property("detection_order", 7, model_info.get('detection_order', ''), is_combobox=True, values=["Ascending X", "Descending X", "Ascending Y", "Descending Y", "Maximum Matching %"])
        self.add_property("detection_count", 8, model_info.get('detection_count', ''))

        self.add_picking_condition(model_info, start_row=9)
        self.add_musk_properties(model_info, start_row=13)

    def add_property(self, label_key, row, value, is_combobox=False, values=None):
        label = tk.Label(self.properties_panel.properties_frame, text=language.translate(label_key) + " :", bg='white')
        label.grid(row=row, column=0, sticky='w', padx=5, pady=5)
        if is_combobox:
            entry = ttk.Combobox(self.properties_panel.properties_frame, values=values, width=18, state='readonly')
            entry.set(value)
        else:
            entry = tk.Entry(self.properties_panel.properties_frame, highlightthickness=1, highlightbackground="black")
            entry.insert(0, value)
            entry.config(validate="key", validatecommand=(self.properties_panel.root.register(self.properties_panel.validate_number), '%P'))
        entry.grid(row=row, column=1, padx=5, pady=5)
        setattr(self, f"{label_key}_entry", entry)

    def add_picking_condition(self, model_info, start_row):
        picking_label = tk.Label(self.properties_panel.properties_frame, text=language.translate("picking_condition"), bg='white', font=("Helvetica", 12, 'bold'))
        picking_label.grid(row=start_row, column=0, columnspan=2, pady=10)

        self.add_property("picking_point", start_row + 1, model_info.get('picking_point', ''), is_combobox=True, values=["2", "3", "4"])
        self.add_property("picking_point_shape", start_row + 2, model_info.get('picking_point_shape', ''), is_combobox=True, values=[language.translate("rectangle"), language.translate("circle")])
        self.add_property("width_picking", start_row + 3, model_info.get('width_picking', ''))
        self.add_property("height_picking", start_row + 4, model_info.get('height_picking', ''))
        self.add_property("radius_picking", start_row + 5, model_info.get('radius_picking', ''))

    def add_musk_properties(self, model_info, start_row):
        musk_label = tk.Label(self.properties_panel.properties_frame, text=language.translate("musk"), bg='white', font=("Helvetica", 12, 'bold'))
        musk_label.grid(row=start_row, column=0, columnspan=2, pady=10)

        self.add_property("select_shape", start_row + 1, model_info.get('shape', ''), is_combobox=True, values=[language.translate("circle"), language.translate("rectangle"), language.translate("polygon"), language.translate("ring")])
        self.add_property("musk_width", start_row + 2, model_info.get('musk_width', ''))
        self.add_property("musk_height", start_row + 3, model_info.get('musk_height', ''))
        self.add_property("musk_angle", start_row + 4, model_info.get('musk_angle', ''))
