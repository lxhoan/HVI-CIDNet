import torch
from torchvision import transforms
from PIL import Image
from net.HVI_transform import RGB_HVI
# Load image → tensor [1, 3, H, W] in [0, 1]
pil_img = Image.open("./input_hf/Cyberpunk-2077-No-Vacancy-e1570482474134.jpg").convert("RGB")
rgb = transforms.ToTensor()(pil_img).unsqueeze(0)  # [1, 3, H, W]
rgb = rgb.cpu()
transform = RGB_HVI().cpu()
transform.eval()  # no effect here, but good habit
with torch.no_grad():
    hvi = transform.HVIT(rgb)
print(hvi.shape)   # torch.Size([1, 3, H, W])
print(hvi.min(), hvi.max())
# Optional: RGB → HVI → RGB round-trip
rgb_back = transform.PHVIT(hvi)  # PHVIT uses this_k from the last HVIT call

rgb_back = torch.clamp(rgb_back, 0, 1)
transforms.ToPILImage()(rgb_back.squeeze(0)).save("RGB_HVI_rgb_back.png")
