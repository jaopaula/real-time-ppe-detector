import torch
import subprocess

def get_gpu_name():
    try:
        output = subprocess.check_output(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        ).decode().strip().split("\n")[1]
        return output.strip()
    except:
        return "Não foi possível identificar a GPU via WMIC."

print("=== TESTE GPU / CUDA / TORCH ===")

print(f"Torch version: {torch.__version__}")
print(f"CUDA disponível (torch.cuda.is_available()): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU detectada pelo Torch: {torch.cuda.get_device_name(0)}")
else:
    print("Torch NÃO detectou GPU.")

print(f"GPU detectada via WMIC: {get_gpu_name()}")
print("================================")