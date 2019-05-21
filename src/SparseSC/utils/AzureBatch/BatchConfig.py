"""
configuration for azure batch
"""
from typing import NamedTuple, Optional
from jsonschema import validate
from .constants import _DOCKER_CONTAINER
import os

# ------------------------------
# Fail Faster
# ------------------------------
config_schema = {
    "type": "object",
    "properties": {
        "BATCH_ACCOUNT_NAME": {"type": "string"},
        "BATCH_ACCOUNT_KEY": {"type": "string"},
        "BATCH_ACCOUNT_URL": {"type": "string"},
        "STORAGE_ACCOUNT_NAME": {"type": "string"},
        "STORAGE_ACCOUNT_KEY": {"type": "string"},
        "REGISTRY_SERVER": {"type": "string"},
        "REGISTRY_USERNAME": {"type": "string"},
        "REGISTRY_PASSWORD": {"type": "string"},
        "POOL_ID": {"type": "string"},
        "POOL_NODE_COUNT": {"type": "number", "minimum": 0},
        "POOL_LOW_PRIORITY_NODE_COUNT": {"type": "number", "minimum": 0},
        "POOL_VM_SIZE": {"type": "string"},
        "DELETE_POOL_WHEN_DONE": {"type": "boolean"},
        "JOB_ID": {"type": "string"},
        "DELETE_JOB_WHEN_DONE": {"type": "boolean"},
        "CONTAINER_NAME": {
            "type": "string",
            "pattern": "^[a-z0-9](-?[a-z0-9]+)$",
            "maxLength": 63,
            "minLength": 3,
        },
        "BATCH_DIRECTORY": {"type": "string"},
        "DOCKER_CONTAINER": {"type": "string"},
    },
    # TODO: missing required properties
}


class BatchConfig(NamedTuple):
    """
    A convenience class for typing the config object
    """

    # pylint: disable=too-few-public-methods
    POOL_ID: str
    JOB_ID: str
    POOL_VM_SIZE: str
    CONTAINER_NAME: str
    BATCH_DIRECTORY: str
    POOL_NODE_COUNT: int = 0
    POOL_LOW_PRIORITY_NODE_COUNT: int = 0
    DELETE_POOL_WHEN_DONE: bool = False
    DELETE_JOB_WHEN_DONE: bool = False
    BATCH_ACCOUNT_NAME: Optional[str] = None
    BATCH_ACCOUNT_KEY: Optional[str] = None
    BATCH_ACCOUNT_URL: Optional[str] = None
    STORAGE_ACCOUNT_NAME: Optional[str] = None
    STORAGE_ACCOUNT_KEY: Optional[str] = None
    REGISTRY_SERVER: Optional[str] = None
    REGISTRY_USERNAME: Optional[str] = None
    REGISTRY_PASSWORD: Optional[str] = None
    DOCKER_CONTAINER: Optional[str] = _DOCKER_CONTAINER


service_keys = (
    "BATCH_ACCOUNT_NAME",
    "BATCH_ACCOUNT_KEY",
    "BATCH_ACCOUNT_URL",
    "STORAGE_ACCOUNT_NAME",
    "STORAGE_ACCOUNT_KEY",
    "REGISTRY_SERVER",
    "REGISTRY_USERNAME",
    "REGISTRY_PASSWORD",
)

_env_config = {}
for key in service_keys:
    val = os.getenv(key, None)
    if val:
        _env_config[key] = val.strip('"')


def validate_config(config):
    """
    validate the batch configuration object
    """
    _config = config._asdict()
    for _key in service_keys:
        if not _config[_key]:
            del _config[_key]

    __env_config = _env_config.copy()
    __env_config.update(_config)
    validate(__env_config, config_schema)
    return BatchConfig(**__env_config)
