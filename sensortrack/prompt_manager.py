import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape, StrictUndefined, TemplateError

class PromptManager:
    def __init__(self, template_dir: str | Path = '.'):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(),
            undefined=StrictUndefined  # Raise an error for undefined variables
        )

    def render(self, template_name: str, **kwargs) -> str:
        try:
            template = self.env.get_template(template_name)
        except TemplateError as e:
            raise RuntimeError(f"Error loading template '{template_name}': {e}") from e

        try:
            return template.render(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Error rendering template '{template_name}' with parameters {kwargs}: {e}"
            ) from e 