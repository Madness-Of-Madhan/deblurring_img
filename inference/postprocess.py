import numpy as np
import base64
import io
from PIL import Image

def tensor_to_base64(tensor):
    if tensor.dim() == 4:  # batch size 1
        tensor = tensor[0]

    tensor = tensor.detach().cpu()
    img = tensor.permute(1, 2, 0).numpy()
    img = (img + 1) / 2
    img = (img * 255).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")
