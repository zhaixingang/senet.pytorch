import torch
from thop import profile
from senet.se_resnet import se_resnet50


def main():
    senet = se_resnet50(num_classes=1000)
    senet.load_state_dict(torch.load("./weights/seresnet50-60a8950a85b2b.pkl"))

    senet.eval()
    senet.to("cpu")

    model_path = f"./onnx/senet.onnx"

    dummy_input = torch.randn(1, 3, 224, 224).to("cpu")
    flops, _ = profile(senet, inputs=(dummy_input, ))
    print(flops)
    torch.onnx.export(senet, dummy_input, model_path, verbose=False, input_names=['input'],
                      output_names=['output'])


if __name__ == "__main__":
    main()
