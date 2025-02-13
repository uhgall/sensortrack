from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
import sys
import webbrowser
import argparse

class TimeSeriesPlotter:
    @staticmethod
    def get_parser(description: str = "Analyze and plot sensor data from processed images.") -> argparse.ArgumentParser:
        """Create a command line argument parser with common options."""
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Example usage:
  python script.py path/to/data/dir

Directory structure should be:
  data_dir/
  ├── *.jpg             # Original images
  └── results/          # JSON files and analysis output
      ├── *.json       # Processed readings
      ├── index.html   # Interactive plot
      └── readings.csv # Data export
"""
        )
        parser.add_argument('working_dir', nargs='?', default='.',
                           help='Directory containing the images and results (default: current directory)')
        return parser

    @classmethod
    def run_from_args(cls, fields_to_plot: List[str], y_label: str = "Value", description: str | None = None, events: List[Tuple[int, str]] | None = None) -> None:
        """Run the plotter from command line arguments."""
        parser = cls.get_parser(description or "Analyze and plot sensor data from processed images.")
        args = parser.parse_args()
        
        plotter = cls(args.working_dir)
        plotter.analyze_and_plot(fields_to_plot, y_label=y_label, events=events)

    def __init__(self, working_dir: str | Path = '.'):
        """Initialize plotter with path to working directory."""
        self.working_dir = Path(working_dir)
        if not self.working_dir.exists():
            print(f"\nError: Directory not found: {self.working_dir}")
            sys.exit(1)
            
        self.results_dir = self.working_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        self.image_number_for_timestamp = {}  # Maps image number (int) -> timestamp

    def _get_color_palette(self, num_colors: int) -> Dict[str, str]:
        """Get a color palette with distinct, visually pleasing colors.
        Uses plotly's qualitative color sequences, cycling through them if needed."""
        # Combine multiple plotly qualitative color sequences for more options
        all_colors = (
            plotly.colors.qualitative.Set2 +
            plotly.colors.qualitative.Pastel2 +
            plotly.colors.qualitative.Set3
        )
        
        # Cycle through colors if we need more than available
        return {str(i): all_colors[i % len(all_colors)] for i in range(num_colors)}

    def analyze_and_plot(self, fields_to_plot: List[str], y_label: str = "Value",
                        open_in_browser: bool = True, events: List[Tuple[int, str]] | None = None) -> None:
        """Main method to analyze data and create visualizations."""
        # Load and process data
        data = self.load_data()
        
        # Generate output file paths
        output_file = (self.results_dir / 'index.html').absolute()
        csv_file = (self.results_dir / 'readings.csv').absolute()
        
        # Generate color palette
        colors = self._get_color_palette(len(fields_to_plot))
        field_colors = dict(zip(fields_to_plot, colors.values()))
        
        # Create interactive plot
        self.create_interactive_plot(
            data, 
            fields_to_plot, 
            field_colors, 
            output_file, 
            y_label=y_label,
            events=events
        )
        print(f"Interactive plot saved to {output_file}")
        
        # Save to CSV for further analysis
        self.save_csv(data, str(csv_file))
        print(f"Raw data saved to {csv_file}")
        
        # Automatically open the plot in the default browser
        if open_in_browser:
            print("\nOpening plot in browser...")
            webbrowser.open(f"file://{output_file}")
                


    def load_data(self) -> List[Dict[str, Any]]:
        """Load all JSON files from results directory, sorted by timestamp."""
        if not self.results_dir.exists():
            print(f"\nResults directory not found: {self.results_dir}")
            print("Have you processed the images first? Try running:")
            print("  python process_images.py <directory>")
            sys.exit(1)
            
        data = []
        json_files = list(self.results_dir.glob('*.json'))
        
        if not json_files:
            print(f"\nNo JSON files found in '{self.results_dir}'")
            print("Have you processed the images first? Try running:")
            print("  python process_images.py <directory>")
            sys.exit(1)
        
        print(f"\nFound {len(json_files)} JSON files in {self.results_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    reading = json.load(f)
                    # Convert ISO timestamp string back to datetime
                    reading['timestamp'] = datetime.fromisoformat(reading['timestamp'])
                    # Store source filename
                    reading['_source_file'] = json_file.name
                    data.append(reading)
                    
                    # Build image number -> timestamp mapping
                    filename = json_file.stem
                    if filename.startswith('IMG_'):
                        try:
                            # Convert to int to drop leading zeros
                            img_num = int(filename[4:])  # Skip 'IMG_'
                            self.image_number_for_timestamp[img_num] = reading['timestamp']
                        except (ValueError, IndexError):
                            continue
                            
            except Exception as e:
                print(f"Error loading {json_file}: {str(e)}")
        
        if not data:
            print("\nNo valid readings loaded from JSON files")
            print("The JSON files might be corrupted or in the wrong format")
            sys.exit(1)
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        print(f"Successfully loaded {len(sorted_data)} readings")
        print(f"Available image numbers: {sorted(self.image_number_for_timestamp.keys())}")
        return sorted_data

    def plot_time_series(self, 
                        data: List[Dict[str, Any]], 
                        fields_to_plot: List[str],
                        title: str = "Sensor Readings Over Time",
                        y_label: str = "Value",
                        width: int = 1200,
                        height: int = 1200,
                        show_rangeslider: bool = False,
                        save_path: Optional[str] = None) -> None:
        """Plot time series data for specified fields using Plotly."""
        if not data:
            raise ValueError("No data provided for plotting")
            
        if not fields_to_plot:
            raise ValueError("No fields specified for plotting")
            
        # Verify all fields exist in the data
        sample_reading = data[0]
        missing_fields = [field for field in fields_to_plot if field not in sample_reading]
        if missing_fields:
            raise ValueError(f"Fields not found in data: {missing_fields}")
            
        fig = go.Figure()
        
        timestamps = [d['timestamp'] for d in data]
        print(f"Plotting {len(timestamps)} data points from {timestamps[0]} to {timestamps[-1]}")
        
        for field in fields_to_plot:
            values = [d[field] for d in data]
            valid_values = [(t, v) for t, v in zip(timestamps, values) if v != -1]
            if not valid_values:
                print(f"Warning: No valid readings for {field}")
                continue
                
            valid_timestamps, valid_values = zip(*valid_values)
            print(f"Field {field}: {len(valid_values)} valid readings")
            
            fig.add_trace(
                go.Scatter(
                    x=valid_timestamps,
                    y=valid_values,
                    mode='lines+markers',
                    name=field
                )
            )
        
        if not fig.data:
            raise ValueError("No valid data to plot")
            
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title=y_label,
            width=width,
            height=height,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        if show_rangeslider:
            fig.update_xaxes(rangeslider_visible=True)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
        else:
            fig.show()  # Only show in browser if not saving to file

    def save_csv(self, data: List[Dict[str, Any]], output_file: str | Path) -> None:
        """Save time series data to CSV file."""
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Remove internal fields
        columns_to_drop = ['_source_file']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Reorder columns to put timestamp first if it exists
        columns = []
        if 'timestamp' in df.columns:
            columns.append('timestamp')
        columns.extend([col for col in df.columns if col != 'timestamp'])
        
        # Reorder columns
        df = df[columns]
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Raw data saved to {output_file}")

    def filter_invalid_readings(self, data: List[Dict[str, Any]], field: str) -> List[Tuple[datetime, float, str]]:
        """Filter out readings where a specific sensor has -1 value."""
        return [(d['timestamp'], d[field], Path(d['_source_file']).stem if '_source_file' in d else '') 
                for d in data if d[field] != -1]

    def split_into_continuous_segments(self, timestamps: List[datetime], values: List[float], 
                                    filenames: List[str], max_gap: timedelta = timedelta(minutes=30)) -> List[Tuple[List, List, List]]:
        """Split data into continuous segments, returns list of (timestamps, values, filenames) tuples."""
        if not timestamps:
            return []
        
        segments = []
        current_segment_times = [timestamps[0]]
        current_segment_values = [values[0]]
        current_segment_files = [filenames[0]]
        
        for i in range(1, len(timestamps)):
            time_gap = timestamps[i] - timestamps[i-1]
            if time_gap > max_gap:
                # Gap detected, start new segment
                if current_segment_times:
                    segments.append((current_segment_times, current_segment_values, current_segment_files))
                current_segment_times = [timestamps[i]]
                current_segment_values = [values[i]]
                current_segment_files = [filenames[i]]
            else:
                current_segment_times.append(timestamps[i])
                current_segment_values.append(values[i])
                current_segment_files.append(filenames[i])
        
        if current_segment_times:
            segments.append((current_segment_times, current_segment_values, current_segment_files))
        
        return segments

    def create_interactive_plot(self, data: List[Dict[str, Any]], fields_to_plot: List[str], 
                              colors: Dict[str, str], output_file: Path, y_label: str = "Value",
                              events: List[Tuple[int, str]] | None = None) -> None:
        """Create an interactive plot with click-to-view-image functionality."""
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Plot sensor data first
        for field in fields_to_plot:
            valid_readings = self.filter_invalid_readings(data, field)
            if valid_readings:
                timestamps, values, filenames = zip(*valid_readings)
                segments = self.split_into_continuous_segments(timestamps, values, filenames)
                
                for i, (seg_times, seg_values, seg_files) in enumerate(segments):
                    color = colors.get(field, '#808080')
                    fig.add_trace(
                        go.Scatter(
                            x=seg_times,
                            y=seg_values,
                            mode='lines+markers',
                            name=field if i == 0 else f"{field}_segment_{i+1}",
                            showlegend=i == 0,
                            line=dict(color=color),
                            marker=dict(size=8, color=color),
                            customdata=seg_files,
                            hovertemplate=f"%{{text}}: %{{y:.0f}} {y_label}<extra></extra>",
                            text=[f"{field}" for _ in seg_times]
                        )
                    )

        # Add event annotations if provided
        if events:
            print(f"\nProcessing {len(events)} events...")
            
            # Calculate y-axis range
            y_min = float('inf')
            y_max = float('-inf')
            for trace in fig.data:
                if trace.y:  # Check if trace has y values
                    y_min = min(y_min, min(trace.y))
                    y_max = max(y_max, max(trace.y))
            
            # Update layout
            fig.update_layout(
                title=f"Sensor {y_label} Over Time",
                xaxis_title='Time',
                yaxis_title=y_label,
                yaxis=dict(
                    gridcolor='rgba(128, 128, 128, 0.1)',
                    dtick=100,  # Regular gridlines every 100 units
                    gridwidth=1,
                    tick0=0,
                    showgrid=True,
                    range=[y_min, y_max + (y_max - y_min) * 0.05]  # Add 5% padding at the top
                ),
                width=1200,
                height=1200,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                    namelength=-1
                ),
                hoverlabel_align='left',
                clickmode='event+select'
            )
            
            # Add thicker gridlines at 1000-unit intervals
            y_start = (y_min // 1000) * 1000
            y_end = ((y_max // 1000) + 1) * 1000
            
            shapes = []
            for y in range(int(y_start), int(y_end) + 1000, 1000):
                shapes.append(dict(
                    type="line",
                    xref="paper",
                    yref="y",
                    x0=0,
                    x1=1,
                    y0=y,
                    y1=y,
                    line=dict(
                        color="rgba(128, 128, 128, 0.5)",
                        width=2,
                    ),
                    layer="below"
                ))
            fig.update_layout(shapes=shapes)
            
            # Add range slider
            fig.update_xaxes(rangeslider_visible=True)
            
            # Add events
            for img_num, description in events:
                if img_num not in self.image_number_for_timestamp:
                    print(f"Warning: Could not find data for image number {img_num}")
                    continue
                    
                timestamp = self.image_number_for_timestamp[img_num]
                print(f"Adding event: {description} at image {img_num}")
                
                # Add vertical line
                fig.add_shape(
                    type="line",
                    x0=timestamp,
                    x1=timestamp,
                    y0=y_min,
                    y1=y_max,
                    line=dict(
                        color="rgba(128, 128, 128, 0.3)",
                        width=2,
                        dash="solid"
                    )
                )
                
                # Add annotation
                fig.add_annotation(
                    x=timestamp,
                    y=y_max + (y_max - y_min) * 0.05,
                    text=description,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="rgba(128, 128, 128, 0.5)",
                    ax=0,
                    ay=-40,
                    font=dict(size=10),
                    textangle=-45
                )
        
        # Save the plot with click handling JavaScript
        with open(output_file, 'w') as f:
            # Write the plot HTML
            plot_html = fig.to_html(
                include_plotlyjs=True,
                full_html=True,
                default_width='100%',
                default_height='100%'
            )
            
            # Add click handling JavaScript
            click_handler = """
            <script>
            var plot = document.getElementsByClassName('plotly-graph-div')[0];
            plot.on('plotly_click', function(data) {
                if (data.points.length > 0) {
                    var point = data.points[0];
                    if (point.customdata) {
                        window.open('../' + point.customdata + '.jpg', '_blank');
                    }
                }
            });
            </script>
            """
            
            # Insert the click handler before the closing body tag
            plot_html = plot_html.replace('</body>', f'{click_handler}</body>')
            f.write(plot_html)

def plot_sensor_data(
    df: pd.DataFrame,
    title: str = "Sensor Readings",
    width: int = 1200,
    height: int = 1200,
    ylabel: str = "Value",
    show_rangeslider: bool = False,
    save_path: Optional[str] = None
) -> None:
    """
    Plot time series sensor data using Plotly.
    
    Args:
        df: DataFrame with datetime index and values
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        ylabel: Y-axis label
        show_rangeslider: Whether to show the range slider
        save_path: Optional path to save the plot as HTML
    """
    fig = go.Figure()
    
    # Handle both single column and multiple column DataFrames
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    for column in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                mode='lines+markers',
                name=column
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=ylabel,
        width=width,
        height=height,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    if show_rangeslider:
        fig.update_xaxes(rangeslider_visible=True)
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()  # Only show in browser if not saving to file 