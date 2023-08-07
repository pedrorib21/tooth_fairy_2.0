import vtk

def read_obj(filename:str)-> vtk.vtkPolyData:
    objReader = vtk.vtkOBJReader()
    objReader.SetFileName(filename)
    objReader.Update()
    polydata = objReader.GetOutput()
    return polydata