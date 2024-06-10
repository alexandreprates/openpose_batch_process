#!/usr/bin/env python

import os
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
import sys

#sys.stdout.buffer = None

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
processor.to(device)

path = sys.argv[1]
files = os.listdir(path)
num_files = len(files)

# Print a countdown for each file
for i in range(0, num_files -1):
    file = files[i]
    
    if file.endswith("openpose.png"):
        continue
    
    source_image = f"{path}/{file}"
    destination_image = f"{path}/{file}.openpose2.png"

    print(f"processing {i} of {num_files} - {source_image} -> {destination_image} ", end="")

    image = load_image(source_image)
    control_image = processor(image, hand_and_face=True)
    control_image.save(destination_image)

    print("done!")

