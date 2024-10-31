import sys
sys.path.append("/Applications/FreeCAD.app/Contents/Resources/lib/python3.10/site-packages")

import os
import FreeCAD
import Part

def convert_step_to_stl(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.step') or filename.endswith('.stp'):
            step_file = os.path.join(input_folder, filename)
            stl_file = os.path.join(output_folder, filename.replace('.step', '.stl').replace('.stp', '.stl'))

            # Load the STEP file
            doc = FreeCAD.newDocument()
            shape = Part.Shape()
            shape.read(step_file)

            # Export to STL
            Part.export([shape], stl_file)
            print(f"Converted {step_file} to {stl_file}")

# Example usage
input_folder = os.path.join(os.path.dirname(__file__), '../proj_log/pretrained/results/test_1000_step/')
output_folder = os.path.join(os.path.dirname(__file__), '../proj_log/pretrained/results/test_1000_stl/')
convert_step_to_stl(input_folder, output_folder)

