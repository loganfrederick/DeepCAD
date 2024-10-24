import os
import glob
import json
import h5py
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
import argparse
import sys
from OCC.Core.IFSelect import IFSelect_RetDone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD
from file_utils import ensure_dir


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")
parser.add_argument('--form', type=str, default="h5", choices=["h5", "json"], help="file format")
parser.add_argument('--idx', type=int, default=0, help="export n files starting from idx.")
parser.add_argument('--num', type=int, default=10, help="number of shapes to export. -1 exports all shapes.")
parser.add_argument('--filter', action="store_true", help="use opencascade analyzer to filter invalid model")
parser.add_argument('-o', '--outputs', type=str, default=None, help="save folder")
args = parser.parse_args()

src_dir = args.src
print(src_dir)
out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format(args.form))))
if args.num != -1:
    out_paths = out_paths[args.idx:args.idx+args.num]
save_dir = args.src + "_step" if args.outputs is None else args.outputs
ensure_dir(save_dir)
print(f"Output directory: {os.path.abspath(save_dir)}")

print(f"Files to process: {out_paths}")

for path in out_paths:
    print(f"Processing: {os.path.abspath(path)}")
    try:
        if args.form == "h5":
            with h5py.File(path, 'r') as fp:
                out_vec = fp["out_vec"][:].astype(float)
                out_shape = vec2CADsolid(out_vec)
        else:
            with open(path, 'r') as fp:
                data = json.load(fp)
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            out_shape = create_CAD(cad_seq)

        print(f"out_shape created: {out_shape is not None}")

        if args.filter:
            analyzer = BRepCheck_Analyzer(out_shape)
            if not analyzer.IsValid():
                print("Detected invalid model.")
                continue
        
        name = path.split("/")[-1].split(".")[0]
        save_path = os.path.join(save_dir, name + ".step")
        print(f"Attempting to save to: {os.path.abspath(save_path)}")
        
        writer = STEPControl_Writer()
        transfer_status = writer.Transfer(out_shape, STEPControl_AsIs)
        if transfer_status != IFSelect_RetDone:
            print(f"Failed to transfer shape for {save_path}")
            continue
        
        write_status = writer.Write(save_path)
        if write_status == IFSelect_RetDone:
            print(f"Successfully wrote: {save_path}")
        else:
            print(f"Failed to write {save_path}")

    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        continue

print(f"Processing complete. Check {save_dir} for output files.")
