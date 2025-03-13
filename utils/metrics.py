import math

from skimage.metrics import structural_similarity


def compute_ssim(image_outputs, image_sources):
    """
    SSIM
    """
    ssim_list = []
    if image_outputs.shape != image_sources.shape or image_outputs.shape[0] != image_sources.shape[0]:
        raise AssertionError("Image outputs and image sources shape mismatch.")
    outputs_np = ((image_outputs + 1) / 2).clamp(0, 1).cpu().detach().numpy()
    sources_np = ((image_sources + 1) / 2).clamp(0, 1).cpu().detach().numpy()

    outputs_np = outputs_np.transpose(0, 2, 3, 1)
    sources_np = sources_np.transpose(0, 2, 3, 1)
    length = image_outputs.shape[0]
    for output, source in zip(outputs_np, sources_np):
        ssim = structural_similarity(
            output,
            source,
            channel_axis=-1,  # 通道在最后一维
            data_range=1.0  # 数据范围 [0,1]
        )
        ssim_list.append(ssim)
    return sum(ssim_list) / length

def compute_psnr(mse):
    """
    PSNR
    """
    # Results
    if mse == 0:
        return 100
    else:
        return 20 * math.log10(255.0 / math.sqrt(mse))