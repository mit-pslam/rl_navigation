project('rl_navigation_bindings', 'cpp', version: '0.0.1',
  default_options : ['cpp_std=c++17', 'warning_level=2', 'b_ndebug=if-release'])

pybind11_project = subproject('pybind11')
pybind11_dep = pybind11_project.get_variable('pybind11_dep')

cmake = import('cmake')
sophus_project = cmake.subproject('sophus', cmake_options: ['-DBUILD_TESTS=OFF', '-DBUILD_EXAMPLES=OFF', '-DSOPHUS_USE_BASIC_LOGGING=ON'])
sophus_dep = sophus_project.dependency('sophus')

eigen_dep = dependency('Eigen3', method: 'cmake', modules: ['Eigen3::Eigen'])
thread_dep = dependency('threads')

build_result_path = get_option('build_result_path')
interpreter_path = get_option('interpreter_path')

pymod = import('python')

py = pymod.find_installation(interpreter_path)
install_interp = find_program(interpreter_path)

py_dep = py.dependency()

subdir('sophus')
