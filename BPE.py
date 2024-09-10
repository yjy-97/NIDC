import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def read_nii(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return torch.tensor(data, dtype=torch.float32)

def get_brain_areas(data):
    areas = torch.unique(data)
    areas = areas[areas != 0]
    return areas

def process_brain_area(data, area_label):
    mask = (data == area_label).float()

    distance_transform = torch.tensor(distance_transform_edt(mask.cpu().numpy()), dtype=torch.float32)
    max_distance = torch.max(distance_transform)

    area_processed = torch.zeros_like(data)

    if max_distance == 0:
        return area_processed

    edge_points = (distance_transform == 1).float()

    center_points = (distance_transform == max_distance).float()

    area_processed[edge_points.bool()] = 1 if area_label % 2 == 0 else -1
    area_processed[center_points.bool()] = 0

    intermediate_points = (distance_transform > 1) & (distance_transform < max_distance)
    area_processed[intermediate_points] = (max_distance - distance_transform[intermediate_points]) / max_distance

    if area_label % 2 != 0:
        area_processed[intermediate_points] *= -1

    return area_processed


def process_template(data):
    brain_areas = get_brain_areas(data)
    processed_data = torch.zeros_like(data)

    for area in brain_areas:
        processed_area = process_brain_area(data, area)
        processed_data += processed_area

    return processed_data

file_path = r''
template_data = read_nii(file_path)

processed_template = process_template(template_data)

output_img = nib.Nifti1Image(processed_template.numpy(), affine=np.eye(4))
nib.save(output_img, r'')
