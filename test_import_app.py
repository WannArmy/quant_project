import importlib.util, sys, pathlib, traceback
spec = importlib.util.spec_from_file_location('app', r'c:\Users\Richie Chua\Downloads\quant_project\app.py')
module = importlib.util.module_from_spec(spec)
sys.modules['app'] = module
try:
    spec.loader.exec_module(module)
    print('App imported successfully')
except Exception as e:
    print('Error importing app:')
    traceback.print_exc()
