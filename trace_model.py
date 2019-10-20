import torch
from torchvision.models import resnet18

model = resnet18()
model.eval()
example = torch.rand(1, 3, 224, 224)

traced_script_module = torch.jit.trace(model, example)
traced_script_module.save('/tmp/resnet18_float32.pt')


model = resnet18()
model.eval()
example = torch.rand(1, 3, 224, 224)

model = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save('/tmp/resnet18_qint8.pt')