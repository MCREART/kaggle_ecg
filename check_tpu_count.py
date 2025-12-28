import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import os

print(f"PJRT_DEVICE: {os.environ.get('PJRT_DEVICE')}")
try:
    devices = xm.get_xla_supported_devices()
    print(f"Supported devices: {len(devices)}")
    print(f"Devices: {devices}")
except Exception as e:
    print(f"Error checking devices: {e}")

print(f"Current device: {xm.xla_device()}")
