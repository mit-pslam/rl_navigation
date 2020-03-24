project('rl_navigation_bindings', 'cpp', version: '0.0.1',
  default_options : ['cpp_std=c++17', 'werror=true', 'warning_level=2', 'b_ndebug=if-release'])

pybind11_project = subproject('pybind11')
pybind11_dep = pybind11_project.get_variable('pybind11_dep')

cmake = import('cmake')
eigen_dep = dependency('Eigen3', method: 'cmake', modules: ['Eigen3::Eigen'])
thread_dep = dependency('threads')

py_install_dir_script = files('scripts/detect_python_install_dir.py')

subdir('bindings')

pymod = import('python')

use_user_sitepackages = get_option('use_user_sitepackages')
python_interp_path = get_option('python_interp_path')

if use_user_sitepackages and python_interp_path != ''
  error('cannot specify user sitepackages install path and an alternative interpreter')
endif

if python_interp_path == ''
  py = pymod.find_installation('python3')
  install_interp = find_program('python3')
else
  py = pymod.find_installation(python_interp_path)
  install_interp = find_program(python_interp_path)
endif

py_dep = py.dependency()

if use_user_sitepackages
  ret = run_command(install_interp, py_install_dir_script, '--user', check : true)
else
  ret = run_command(install_interp, py_install_dir_script, check : true)
endif

py_install_dir = ret.stdout().strip()

subdir('sophus')
