import os
from pathlib import Path
from typing import TypeVar, Generic, Optional, Dict, Any, List, Union, Set, Type
from datetime import datetime
import json
import base64
from PIL import Image, UnidentifiedImageError
import piexif
from pydantic import BaseModel
from .prompt_manager import PromptManager
import logging
from collections import defaultdict
import urllib3

from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_content_part_image_param import ImageURL, ChatCompletionContentPartImageParam
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import pydantic_ai


T = TypeVar('T', bound=BaseModel)

# Update the supported image formats
SUPPORTED_IMAGE_FORMATS: Set[str] = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}

def encode_image_to_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode image {image_path}: {str(e)}")

def get_image_timestamp(image_path: str) -> datetime:
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        if exif_dict and "Exif" in exif_dict:
            if piexif.ExifIFD.DateTimeOriginal in exif_dict["Exif"]:
                date_str = exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
                try:
                    return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    pass  # Fall back to file modification time
    except (UnidentifiedImageError, Exception) as e:
        print(f"Warning: Could not extract EXIF timestamp from {image_path}: {e}")
    
    # Fall back to file modification time
    return datetime.fromtimestamp(os.path.getmtime(image_path))

class SensorConfig:
    """Base class for sensor configuration"""
    def __init__(self, template_dir: str | Path = '.'):
        self.template_dir = template_dir
        self.template_name = 'prompt.txt'  # Fixed template name for all sensors
        self.model_name = "google/gemini-2.0-flash-001"  # Default model

    @property
    def reading_type(self) -> type[BaseModel]:
        """Gets the Reading model from the nested Reading class"""
        return self.Reading  # Each config must define a nested Reading class

    def validate_and_correct(self, current: dict, previous: dict | None) -> dict | None:
        """Override this method to implement sensor-specific validation"""
        return current

class SensorTracker:
    def __init__(self, config: SensorConfig):
        self.config = config
        
        # Set up logging
        self.working_dir = None  # Will be set in process_directory
        self.results_dir = None  # Will be set in process_directory
        
        # Configure logging to both file and stdout
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Add our specific handlers
        formatter = logging.Formatter('%(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler will be added in process_directory when we know the results path
        
        # Prevent logger from propagating to root logger
        self.logger.propagate = False
        
        # Suppress HTTP request logs
        urllib3.disable_warnings()
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
        
        # Set up the AI agent with better error handling
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("Please set OPENROUTER_API_KEY environment variable")

        try:
            model = OpenAIModel(self.config.model_name,
                base_url="https://openrouter.ai/api/v1",  # Fixed URL
                api_key=api_key
            )
            self.agent = Agent(model, result_type=self.config.reading_type)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AI agent: {str(e)}")

    def process_image(self, image_path: str, previous_reading: dict | None = None) -> BaseModel:
        """Process a single image using the AI agent."""
        base64_image = encode_image_to_base64(image_path)
        
        prompt_manager = PromptManager(self.config.template_dir)
        prompt_text = prompt_manager.render(self.config.template_name, previous_reading=previous_reading)
        
        image = ImageURL(
            url=f"data:image/jpeg;base64,{base64_image}",
            detail="low"
        )

        content = [
            ChatCompletionContentPartTextParam(type="text", text=prompt_text),
            ChatCompletionContentPartImageParam(type="image_url", image_url=image)
        ]
        
        result = self.agent.run_sync(content)
        reading_data = result.data
        reading_data.timestamp = get_image_timestamp(image_path)

        # If we have a previous reading, preserve its values for any None fields
        if previous_reading:
            current_data = reading_data.model_dump()
            for key, value in current_data.items():
                if value is None and key in previous_reading:
                    # Don't copy over timestamp from previous reading
                    if key != 'timestamp':
                        setattr(reading_data, key, previous_reading[key])
        
        return reading_data

    def process_directory(self, images_dir: str | Path) -> None:
        """Process all images in a directory."""
        self.working_dir = Path(images_dir)
        if not self.working_dir.exists():
            raise ValueError(f"Directory not found: {self.working_dir}")

        # Set up results directory under the working directory
        self.results_dir = self.working_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Now that we know the results path, add the file handler for logging
        file_handler = logging.FileHandler(self.results_dir / "processing.log")
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        validation_stats = {
            'existing_failed': 0,
            'new_failed': 0,
            'total_processed': 0
        }

        # Get all image files and sort by timestamp
        image_paths = [
            p for p in self.working_dir.glob("*") 
            if p.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        ]
        
        if not image_paths:
            raise ValueError(f"No supported images found in {self.working_dir}. "
                           f"Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}")

        image_paths.sort(key=get_image_timestamp)

        previous_data = None
        for image_path in image_paths:
            json_filename = self.results_dir / f"{image_path.stem}.json"
            current_data = None
            is_existing = False

            # Try to load existing data
            if json_filename.exists():
                try:
                    with open(json_filename) as f:
                        current_data = json.load(f)
                        is_existing = True
                except Exception as e:
                    self.logger.error(f"\nError loading existing file {json_filename}:")
                    self.logger.error(f"Error details:\n{str(e)}")
                    continue

            # Process new image if needed
            if current_data is None:
                try:
                    self.logger.info(f"Processing {image_path.name}...")
                    reading = self.process_image(str(image_path), previous_data)
                    current_data = reading.model_dump()
                    # Convert timestamp to string for JSON
                    if 'timestamp' in current_data:
                        current_data['timestamp'] = current_data['timestamp'].isoformat()
                except Exception as e:
                    self.logger.error(f"\nError processing image {image_path}:")
                    self.logger.error(f"Error details:\n{str(e)}")
                    continue
            else:
                self.logger.info(f"Validating existing results for {image_path.name}...")

            validation_stats['total_processed'] += 1

            # Save the data before validation
            with open(json_filename, 'w') as f:
                json.dump(current_data, f, indent=2)

            # Validate the data
            validation_passed = False
            if self.config.validate_and_correct:
                try:
                    validated_data = self.config.validate_and_correct(current_data, previous_data)
                    if validated_data is not None:
                        validation_passed = True
                        # Save the validated/corrected data back to the file
                        with open(json_filename, 'w') as f:
                            json.dump(validated_data, f, indent=2)
                        current_data = validated_data  # Update current_data with validated version
                    else:
                        with open(self.results_dir / 'validation_failed.csv', 'a') as f:
                            f.write(f"{image_path.stem}.json\n")
                        if is_existing:
                            validation_stats['existing_failed'] += 1
                        else:
                            validation_stats['new_failed'] += 1
                except Exception as e:
                    self.logger.error(f"\nError validating {image_path}:")
                    self.logger.error(f"Error details:\n{str(e)}")
                    with open(self.results_dir / 'validation_failed.csv', 'a') as f:
                        f.write(f"{image_path.stem}.json\n")
                    if is_existing:
                        validation_stats['existing_failed'] += 1
                    else:
                        validation_stats['new_failed'] += 1

            # Always update previous_data with current_data for next prompt
            # This ensures continuity in the time series even if validation fails
            previous_data = current_data
            
            if validation_passed:
                self.logger.info(f"Successfully processed and validated {image_path.name}")
            else:
                self.logger.info(f"Saved {json_filename.name} but validation failed")

        # Print summary
        self.logger.info(f"\nResults saved in {self.results_dir}/")
        total_failed = validation_stats['existing_failed'] + validation_stats['new_failed']
        if total_failed > 0:
            self.logger.info(
                f"\nValidation failures ({total_failed} total):\n"
                f"  {validation_stats['new_failed']} from newly processed images\n"
                f"  {validation_stats['existing_failed']} from existing results\n"
                f"See {self.results_dir}/validation_failed.csv for the list of failed files"
            )

    @classmethod
    def run(cls, config: Type[SensorConfig]) -> None:
        """Main entry point for running the sensor tracker."""
        import argparse
        parser = argparse.ArgumentParser(description='Process sensor images')
        parser.add_argument('images_dir', type=str, help='Path to directory containing sensor images')
        args = parser.parse_args()

        tracker = cls(config())
        tracker.process_directory(args.images_dir) 