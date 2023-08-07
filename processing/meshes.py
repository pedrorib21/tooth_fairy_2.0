import vtk


def define_sphere_source(center: list, radius: float) -> vtk.vtkSphereSource:
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetCenter(center)
    sphere_source.SetRadius(radius)
    sphere_source.SetThetaResolution(100)
    sphere_source.SetPhiResolution(100)
    sphere_source.Update()
    return sphere_source


def define_plane(origin: list, normal: list) -> vtk.vtkPlane:
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)
    return plane


def cut_by_bounds(polydata: vtk.vtkPolyData, bounds: list) -> vtk.vtkPolyData:
    plane1 = vtk.vtkPlane()
    plane1.SetOrigin(bounds[0], 0, 0)
    plane1.SetNormal(-1, 0, 0)

    plane2 = vtk.vtkPlane()
    plane2.SetOrigin(bounds[1], 0, 0)
    plane2.SetNormal(1, 0, 0)

    plane3 = vtk.vtkPlane()
    plane3.SetOrigin(0, bounds[2], 0)
    plane3.SetNormal(0, -1, 0)

    plane4 = vtk.vtkPlane()
    plane4.SetOrigin(0, bounds[3], 0)
    plane4.SetNormal(0, 1, 0)

    plane5 = vtk.vtkPlane()
    plane5.SetOrigin(0, 0, bounds[4])
    plane5.SetNormal(0, 0, -1)

    plane6 = vtk.vtkPlane()
    plane6.SetOrigin(0, 0, bounds[5])
    plane6.SetNormal(0, 0, 1)

    clipFunction = vtk.vtkImplicitBoolean()
    clipFunction.SetOperationTypeToIntersection()
    clipFunction.AddFunction(plane1)
    clipFunction.AddFunction(plane2)
    clipFunction.AddFunction(plane3)
    clipFunction.AddFunction(plane4)
    clipFunction.AddFunction(plane5)
    clipFunction.AddFunction(plane6)

    # Clip it.
    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(clipFunction)
    clipper.SetInputData(polydata)
    clipper.GenerateClipScalarsOff()
    clipper.GenerateClippedOutputOff()
    clipper.Update()
    return clipper.GetOutput()
