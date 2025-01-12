## https://examples.vtk.org/site/Cxx/PolyData/PointInsideObject/


import vtk

def main():
    colors = vtk.vtkNamedColors()

    # A cube centered on the origin, 1cm sides.
    cubeSource = vtk.vtkCubeSource()
    cubeSource.Update()

    cube = cubeSource.GetOutput()

    testInside = [0.0, 0.0, 0.0]
    testOutside = [0.7, 0.0, 0.0]
    testBorderOutside = [0.5, 0.0, 0.0]

    points = vtk.vtkPoints()
    points.InsertNextPoint(testInside)
    points.InsertNextPoint(testOutside)
    points.InsertNextPoint(testBorderOutside)

    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)

    # Points inside test.
    selectEnclosedPoints = vtk.vtkSelectEnclosedPoints()
    selectEnclosedPoints.SetInputData(pointsPolydata)
    selectEnclosedPoints.SetSurfaceData(cube)
    selectEnclosedPoints.Update()

    for i in range(3):
        print(f"Point {i}: ", end="")
        if selectEnclosedPoints.IsInside(i) == 1:
            print("inside")
        else:
            print("outside")

    insideArray = selectEnclosedPoints.GetOutput().GetPointData().GetArray("SelectedPoints")

    for i in range(insideArray.GetNumberOfTuples()):
        print(f"Tuple {i}: ", end="")
        if insideArray.GetComponent(i, 0) == 1:
            print("inside")
        else:
            print("outside")

    # RENDERING PART

    # Cube mapper, actor.
    cubeMapper = vtk.vtkPolyDataMapper()
    cubeMapper.SetInputConnection(cubeSource.GetOutputPort())

    cubeActor = vtk.vtkActor()
    cubeActor.SetMapper(cubeMapper)
    cubeActor.GetProperty().SetOpacity(0.5)
    cubeActor.GetProperty().SetColor(colors.GetColor3d("SandyBrown"))

    # Points mapper, actor.
    # First, apply vtkVertexGlyphFilter to make cells around points, vtk only
    # renders cells.
    vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
    vertexGlyphFilter.AddInputData(pointsPolydata)
    vertexGlyphFilter.Update()

    pointsMapper = vtk.vtkPolyDataMapper()
    pointsMapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())

    pointsActor = vtk.vtkActor()
    pointsActor.SetMapper(pointsMapper)
    pointsActor.GetProperty().SetPointSize(5)
    pointsActor.GetProperty().SetColor(colors.GetColor3d("GreenYellow"))

    # Create a renderer, render window, and interactor.
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetWindowName("PointCellIds")

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actor to the scene.
    renderer.AddActor(cubeActor)
    renderer.AddActor(pointsActor)
    renderer.SetBackground(colors.GetColor3d("SlateGray"))

    # Render and interact.
    renderWindow.Render()
    renderWindowInteractor.Start()

if __name__ == "__main__":
    main()
