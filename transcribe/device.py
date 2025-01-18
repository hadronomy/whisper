import torch


class DeviceManager:
    @staticmethod
    def get_default_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def ensure_same_device(tensor, target_device):
        return tensor.to(target_device) if tensor.device != target_device else tensor
