import os
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
import traceback
from OCC.Core.BRepTools import BRepTools_ShapeSet
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

def convert_step_to_stl(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.step') or filename.endswith('.stp'):
            step_file = os.path.join(input_folder, filename)
            stl_file = os.path.join(output_folder, filename.replace('.step', '.stl').replace('.stp', '.stl'))

            # Read the STEP file
            step_reader = STEPControl_Reader()
            status = step_reader.ReadFile(step_file)

            if status == IFSelect_RetDone:
                step_reader.TransferRoots()
                shape = step_reader.OneShape()

                if shape.IsNull():
                    print(f"Invalid shape for {step_file}")
                    continue

                # Check if the shape is valid
                analyzer = BRepCheck_Analyzer(shape)
                if not analyzer.IsValid():
                    print(f"Shape is not valid for {step_file}")
                    continue

                # Log some basic shape information
                num_faces = 0
                num_edges = 0

                # Count faces
                explorer = TopExp_Explorer(shape, TopAbs_FACE)
                while explorer.More():
                    num_faces += 1
                    explorer.Next()

                # Count edges
                explorer = TopExp_Explorer(shape, TopAbs_EDGE)
                while explorer.More():
                    num_edges += 1
                    explorer.Next()

                print(f"Shape for {step_file} has {num_faces} faces and {num_edges} edges")

                # Configure mesh before STL conversion
                linear_deflection = 0.1
                angular_deflection = 0.5
                mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection)
                mesh.Perform()
                if not mesh.IsDone():
                    print(f"Meshing failed for {step_file}")
                    continue

                # Write to STL
                stl_writer = StlAPI_Writer()
                stl_writer.SetASCIIMode(False)  # Use binary mode for STL

                try:
                    result = stl_writer.Write(shape, stl_file)
                    if result:
                        print(f"Converted {step_file} to {stl_file}")
                    else:
                        print(f"Failed to write {stl_file}. StlAPI_Writer returned False")
                except Exception as e:
                    print(f"Exception occurred while writing {stl_file}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Failed to read {step_file} with status {status}")
        else:
            print(f"Skipping non-STEP file: {filename}")

# Example usage
input_folder = os.path.join(os.path.dirname(__file__), '../proj_log/pretrained/results/test_1000_step/')
output_folder = os.path.join(os.path.dirname(__file__), '../proj_log/pretrained/results/test_1000_stl/')
convert_step_to_stl(input_folder, output_folder)
