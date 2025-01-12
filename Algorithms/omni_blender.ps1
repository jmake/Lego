clear 
## Set environment variables

#$ROOT_PATH=$PSScriptRoot
$ROOT_PATH="C:\Users\zvl_2\AppData\Local\ov\pkg\blender-4.2.0-usd.202.1"
echo "ROOT_PATH:$ROOT_PATH"

$env:PXR_PLUGINPATH_NAME = "$ROOT_PATH\blender_omni_plugin\usd\omniverse\resources;$env:PXR_PLUGINPATH_NAME"
$env:PATH = "$ROOT_PATH\blender_omni_plugin;$env:PATH"
$env:PYTHONPATH = "$ROOT_PATH\blender_omni_plugin\bindings-python;$env:PYTHONPATH"
$env:UMM2_PLUGIN_PATH = "$ROOT_PATH\blender_umm2_plugin"


## Python 
 & "$ROOT_PATH\Release\4.2\python\bin\python.exe" --version 
# & "$ROOT_PATH\Release\4.2\python\bin\python.exe" -m pip list 
# & "$ROOT_PATH\Release\4.2\python\bin\python.exe" -m pip install pymesh2 vtk  
# & "$ROOT_PATH\Release\4.2\python\bin\python.exe" -m pip uninstall  pymesh   
# & "$ROOT_PATH\Release\4.2\python\bin\python.exe" F:\Models\Blender\Scripts\script05a1.py
#exit

## Start Blender
# & "$ROOT_PATH\Release\blender.exe" --help 
# & "$ROOT_PATH\Release\blender.exe" --version 
# & "$ROOT_PATH\Release\blender.exe" --background --python script02a1.py 

# & "$ROOT_PATH\Release\blender.exe" --background --python script03b1.py 

 & "$ROOT_PATH\Release\blender.exe" --background --python F:\Models\Blender\Scripts\script05b1.py
#exit 

  & "$ROOT_PATH\Release\blender.exe" 
##Start-Process -FilePath "$ROOT_PATH\Release\blender.exe" -ArgumentList "--log *usd* --log-level 2"