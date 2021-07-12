import numpy as np
import nibabel as nib
from scipy import ndimage
import engine as eng
import inference as inf
import keras
import tensorrt as trt 

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

serialized_plan_fp32 = 'model.plan'
HEIGHT = 160
WIDTH = 160
DEPTH = 80

def resize_volume(img, dim):
        desired_depth = dim[2]
        desired_width = dim[1]
        desired_height = dim[0]
        # Get current depth
        current_depth = img.shape[-1]
        current_width = img.shape[0]
        current_height = img.shape[1]
        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / 1
        width_factor = 1 / width
        height_factor = 1 / height
        # Rotate
        img = ndimage.rotate(img, -90, reshape=False)
        # Resize across z-axis
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
        return img


img_nii = nib.load('./test.nii.gz')
img = img_nii.get_fdata()

img = img[:,:,20:100]

img_clipped = np.clip(img, -1250, 250)
            
max_value = np.max(img_clipped)
min_value = np.min(img_clipped)
            
img_normalized = (img_clipped - min_value) / (max_value - min_value)

img_resized = resize_volume(img, (160, 160, 80))

print('img_resized shape',img_resized.shape)
#print(np.unique(img_resized))

engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
print('loaded engine')
#h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
inputs, outputs, bindings, stream = inf.allocate_buffers(engine)
print('allocated buffers')
#out = inf.do_inference(engine, img_resized, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH, DEPTH)
out = inf.do_inference(engine.create_execution_context(), bindings, inputs, outputs, stream, batch_size = 1)
print(len(out))
print('output shape from main', out[1].shape)
print(np.unique(out[1]))

output_img = nib.Nifti1Image(out, img_nii.affine)
nib.save(output_img, 'output.nii')




