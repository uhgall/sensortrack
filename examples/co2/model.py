from datetime import datetime
from pydantic import BaseModel

class Reading(BaseModel):
    switchbot: float
    tamtop: float
    asani: float
    ak3: float
    R2: float
    R1: float
    aranet4: float
    timestamp: datetime 