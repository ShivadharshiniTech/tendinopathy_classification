import os
import vtk

GEOM_DIR = r"MobL_ARMS_OpenSim3_bimanual_model/Bimanual Upper Arm Model/Geometry"

def convert_vtp_to_ascii(vtp_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtp_path)
    writer.SetInputData(polydata)
    writer.SetDataModeToAscii()  # This is the key line!
    writer.Write()
    print(f"Converted to ASCII: {os.path.basename(vtp_path)}")

if __name__ == "__main__":
    for fname in os.listdir(GEOM_DIR):
        if fname.lower().endswith('.vtp'):
            convert_vtp_to_ascii(os.path.join(GEOM_DIR, fname))
    print("All .vtp files converted to ASCII format.")
