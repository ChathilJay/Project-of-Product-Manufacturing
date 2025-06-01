import tkinter as tk
import numpy as np
import joblib

# Load the models
feature_prediction_model = joblib.load("feature_predictions.pkl")
quality_classification_model = joblib.load("tuned_rfc_model.pkl")
anomaly_detection_model = joblib.load("anomaly_detection_model.pkl")

# Define optimal setpoints (medians from good quality batches)
optimal_setpoints = {
    "ffte_feed_solids_sp": 50,
    "ffte_production_solids_sp": 42,
    "ffte_steam_pressure_sp": 118.44,
    "tfe_out_flow_sp": 2214.29,
    "tfe_production_solids_sp": 65,
    "tfe_vacuum_pressure_sp": -71.14,
    "tfe_steam_pressure_sp": 120
}

# Define slider ranges (based on dataset min/max)
setpoint_ranges = {
    "ffte_feed_solids_sp": (25, 50),
    "ffte_production_solids_sp": (39, 43),
    "ffte_steam_pressure_sp": (10, 850),
    "tfe_out_flow_sp": (1240, 3278),
    "tfe_production_solids_sp": (0, 98),
    "tfe_vacuum_pressure_sp": (-89, -35),
    "tfe_steam_pressure_sp": (2, 135)
}

setpoint_names = list(setpoint_ranges.keys())

# ------------------------- START GUI CODE -------------------------#
root = tk.Tk()
root.title("Vegemite Production Optimisation Tool")
root.geometry("1280x860")
root.configure(bg='#1b1b1b')

# Dictionaries for variables and labels
setpoint_variables = {}
value_display_labels = {}
difference_labels = {}
offset_labels = {}
manual_entries = {}

# Heading and subheading
tk.Label(root, text="Vegemite Production Optimisation Tool", font=("Helvetica", 26), fg="white", bg="#1b1b1b").pack(pady=(20, 5))
tk.Label(root, text="by Group 3", font=("Helvetica", 16), fg="white", bg="#1b1b1b").pack(pady=(0, 20))

slider_frame = tk.Frame(root, bg="#1b1b1b")
slider_frame.pack(pady=10)

# Create slider blocks for each sp
for index, (name, (min_val, max_val)) in enumerate(setpoint_ranges.items()):
    column = tk.Frame(slider_frame, bg="#1b1b1b")
    column.grid(row=0, column=index, padx=10)

    tk.Label(column, text=name, fg="white", bg="#1b1b1b", font=("Helvetica", 9)).pack()
    var = tk.DoubleVar(value=optimal_setpoints[name]) # Starts with the optimal setpoint value
    entry = tk.Entry(column, width=6, justify='center')
    entry.insert(0, f"{var.get():.1f}")
    entry.pack(pady=(4, 4))

    # Function to update the current setpoint value from manual user input
    def update_manual_entry(event, name=name, var=var, entry=entry):
        try:
            val = float(entry.get()) # Gets the value from the entry box
            val = max(setpoint_ranges[name][0], min(val, setpoint_ranges[name][1])) # Ensure value is within range
            var.set(val) # Update the variable
            entry.delete(0, tk.END) # Clear the entry box
            entry.insert(0, f"{val:.1f}") # Reinsert the chosen value in the entry box
            value_display_labels[name].config(text=f"{val:.1f}") # Update the display label
        except ValueError: # Invalid input error handling
            pass

    entry.bind("<Return>", update_manual_entry) # Bind to Return key
    entry.bind("<FocusOut>", update_manual_entry) # Bind to focus out event (clicking away)

    # Function to sync slider and labels with the variable
    def sync_slider_and_labels(var=var, entry=entry, name=name):
        val = var.get() # Get the current value from the variable
        entry.delete(0, tk.END) # Clear the entry box
        entry.insert(0, f"{val:.1f}") # Reinsert the value in the entry box
        value_display_labels[name].config(text=f"{val:.1f}") # Update the display label
        optimal = optimal_setpoints[name]  # Get the optimal value
        difference = (val - optimal) / abs(optimal) * 100 # Calculate percentage difference between current and optimal
        sign = "+" if difference >= 0 else "" # Determine sign for display based on difference
        difference_labels[name].config(text=f"{sign}{difference:.1f}%") # Update the difference label
        offset_labels[name].config(text="off optimal")

    var.trace_add("write", lambda *args, var=var, entry=entry, name=name: sync_slider_and_labels(var, entry, name)) # Trace changes to the variable

    slider = tk.Scale(column, from_=min_val, to=max_val, orient=tk.VERTICAL, length=350, resolution=(max_val - min_val) / 1000, variable=var, bg="#202020", fg="white", troughcolor="#00C9FC", highlightthickness=0, sliderrelief=tk.FLAT) 
    slider.pack()

    main_sp_label = tk.Label(column, text=f"{var.get():.1f}", fg="cyan", bg="#1b1b1b")
    main_sp_label.pack(pady=(4, 0))
    percent_label = tk.Label(column, text="Δ: +0.0%", fg="lightgray", bg="#1b1b1b")
    percent_label.pack()
    offset_label = tk.Label(column, text="off optimal", fg="lightgray", bg="#1b1b1b", font=("Helvetica", 8))
    offset_label.pack()

    setpoint_variables[name] = var
    value_display_labels[name] = main_sp_label
    difference_labels[name] = percent_label
    offset_labels[name] = offset_label
    manual_entries[name] = entry

quality_label = tk.Label(root, text="Predicted Batch Quality: ?", font=("Helvetica", 18), fg="white", bg="#1b1b1b")
quality_label.pack(pady=30)

anomaly_label = tk.Label(root, text="Anomaly Status: ?", font=("Helvetica", 14), fg="white", bg="#1b1b1b")
anomaly_label.pack(pady=10)

def predict():
    input_values = []
    for name in setpoint_names:
        val = setpoint_variables[name].get()
        input_values.append(val)
        value_display_labels[name].config(text=f"{val:.1f}")
        optimal = optimal_setpoints[name]
        difference = (val - optimal) / abs(optimal) * 100
        sign = "+" if difference >= 0 else ""
        difference_labels[name].config(text=f"{sign}{difference:.1f}%")
        offset_labels[name].config(text="off optimal")

    try:
        base_inputs = np.array([input_values])
        predicted_features = feature_prediction_model.predict(base_inputs)
        full_feature_input = np.concatenate([base_inputs, predicted_features], axis=1)

        predicted_quality = quality_classification_model.predict(full_feature_input)[0]
        quality_label.config(text=f"Predicted Quality: {predicted_quality}")

        anomaly_result = anomaly_detection_model.predict(full_feature_input)[0]
        if anomaly_result == -1:
            anomaly_label.config(text="Anomaly Risk Detected!", fg="red")
        else:
            anomaly_label.config(text="No Anomaly Risk Detected!", fg="green")

    except Exception as error:
        quality_label.config(text="Prediction Error", fg="#FFE600")
        anomaly_label.config(text=f"Error: {str(error)}", fg="#FFE600")

# Prediction button
predict_button = tk.Button(root, text="Predict", command=predict, font=("Helvetica", 12), bg="#202020", fg="white")
predict_button.pack(pady=10)

root.mainloop()
