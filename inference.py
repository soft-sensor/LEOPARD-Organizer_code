"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
import os
import json
from glob import glob
import SimpleITK
import numpy

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")



def run():
    # Read the input

    print("paths list",os.listdir(INPUT_PATH))
    print("paths list images",os.listdir(INPUT_PATH/"images"))
    print("paths list pwsi",os.listdir(INPUT_PATH/"images/prostatectomy-wsi"))
    print("paths list pwsi",os.listdir(INPUT_PATH/"images/prostatectomy-tissue-mask"))
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_array(
        location=INPUT_PATH / "images/prostatectomy-wsi",
    )

    input_prostatectomy_tissue_mask = load_image_file_as_array(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
    )
    
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
        print(f.read())

    # For now, let us set make bogus predictions
    output_overall_survival_years = 42

    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "overall-survival-years.json",
        content=output_overall_survival_years
    )
    
    return 0


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha")) + glob(str(location / "*.tif"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
