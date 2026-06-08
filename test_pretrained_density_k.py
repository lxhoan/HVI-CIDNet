import torch
from torchvision import transforms
from net.CIDNet import CIDNet
from PIL import Image

model = CIDNet().cpu();
model.load_state_dict(torch.load("weights/generalization.pth", map_location="cpu"))
model.eval()

# Load image → tensor [1, 3, H, W] in [0, 1]
pil_img = Image.open("./input_hf/Cyberpunk-2077-No-Vacancy-e1570482474134.jpg").convert("RGB")
rgb = transforms.ToTensor()(pil_img).unsqueeze(0)  # [1, 3, H, W]
rgb = rgb.cpu()

model.trans
hvi = model.trans.HVIT(rgb)
print(hvi.shape)   # torch.Size([1, 3, H, W])
print(hvi.min(), hvi.max())
# Optional: RGB → HVI → RGB round-trip
rgb_back = model.trans.PHVIT(hvi)  # PHVIT uses this_k from the last HVIT call

rgb_back = torch.clamp(rgb_back, 0, 1)
transforms.ToPILImage()(rgb_back.squeeze(0)).save("test_pretrained_density_k_rgb_back.png")

h, v, i = hvi.squeeze(0)  # each is [H, W]
transforms.ToPILImage()(h.unsqueeze(0).clamp(0, 1)).save("hvi_h.png")
transforms.ToPILImage()(v.unsqueeze(0).clamp(0, 1)).save("hvi_v.png")
transforms.ToPILImage()(i.unsqueeze(0).clamp(0, 1)).save("hvi_intensity.png")
