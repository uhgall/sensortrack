from sensortrack.time_series_plotter import TimeSeriesPlotter
from model import Reading

def get_events():
    """Return list of (image_number, description) pairs for timeline events"""
    return [(x + 6107, y) for x, y in [
        (18, "Added 50ml CO2"),
        (39, "Added 50ml CO2"),
        (141, "Added 50ml CO2"),
        (150, "Box repositioned"),
        (153, "Added 50ml CO2"),
        (175, "Added 50ml CO2"),
        (182, "Added 50ml CO2"),
        (190, "Added 100ml CO2"),
    ]]

def main():
    """Analyze and plot CO2 sensor data from generated JSON files"""
    # Plot all CO2 sensor readings - use fields from the model
    fields_to_plot = [f for f in Reading.model_fields.keys() if f != 'timestamp']
    
    # Run plotter with CO2-specific configuration
    TimeSeriesPlotter.run_from_args(
        fields_to_plot,
        y_label="CO2 (ppm)",
        description="Analyze and plot CO2 sensor data from processed images.",
        events=get_events()  # Pass the timeline events
    )

if __name__ == "__main__":
    main() 