import torch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="mobilenet_v2", 
    in_channels=3, 
    classes=1, 
    activation=None
)
model.load_state_dict(torch.load("/home/abhi/peer_ws/src/peer/models/segmentation/mobilenet.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 256, 256) 

torch.onnx.export(
    model,
    dummy_input,
    "mobilenet.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    do_constant_folding=True,
)

print("MobileNet segmentation model exported to ONNX (256x256 input).")
