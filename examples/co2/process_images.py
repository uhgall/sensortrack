import sys
from pathlib import Path
from sensortrack import SensorTracker
from sensortrack.sensor_track import SensorConfig
from datetime import datetime
from dotenv import load_dotenv
from model import Reading

class CO2SensorConfig(SensorConfig):
    """Configuration for processing CO2 sensor images"""
    Reading = Reading  # Use the shared Reading model

    def validate_and_correct(self, current: dict, previous: dict | None, max_ratio: float = 8) -> dict | None:
        """Validate and potentially correct readings. Returns None if fatal error."""
        try:
            # Create a corrected copy of current readings
            corrected = current.copy()
            
            # Validate against previous reading if available
            if previous:
                for field_name, value in current.items():
                    if field_name == 'timestamp':
                        continue
                        
                    prev_value = previous[field_name]
                    
                    # If value is less than 1/8th of previous, multiply by 10 (likely missing digit)
                    if  value / prev_value < 1/max_ratio:
                        corrected_value = value * 10
                        print(f"Auto-correcting '{field_name}': {value} -> {corrected_value} PPM (likely missing digit)")
                        corrected[field_name] = corrected_value
                    
            return corrected

        except Exception as e:
            print(f"Error in validation: {e}")
            return None

if __name__ == "__main__":
    load_dotenv()
    SensorTracker.run(CO2SensorConfig) 