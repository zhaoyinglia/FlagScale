
from typing import Dict

from .base import BaseTaskHandler
from .vlm import VLMHandler
from .t2i import T2IHandler


TASK_REGISTRY: Dict[str, "BaseTaskHandler"] = {}


def register_task(name, cls):
    TASK_REGISTRY[name] = cls


register_task("vlm_sft", VLMHandler)
register_task("t2i", T2IHandler)
