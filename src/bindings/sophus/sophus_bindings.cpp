#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <sophus/so2.hpp>
#include <sophus/so3.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

namespace py = pybind11;

typedef Sophus::SO2<double> SO2Type;
typedef Sophus::SO3<double> SO3Type;
typedef Sophus::SE2<double> SE2Type;
typedef Sophus::SE3<double> SE3Type;

// inspired paritally by:
// https://github.com/craigstar/SophusPy/blob/master/sophus/python/so3.h

PYBIND11_MODULE(sophus, m) {
  m.doc() = "bindings for the sophus lie algebra library";

  py::class_<SO2Type>(m, "SO2")
      .def(py::init<>())
      .def(py::init<double>())
      .def(py::init<const SO2Type&>())
      .def(py::init<const Eigen::Matrix2d&>())
      .def("adjoint", &SO2Type::Adj)
      .def("inverse", &SO2Type::inverse)
      .def("log", &SO2Type::log)
      .def("normalize", &SO2Type::normalize)
      .def("matrix", &SO2Type::matrix)
      .def("copy", [](const SO2Type& so2) { return SO2Type(so2); })
      .def(py::self * py::self)
      .def(py::self * Eigen::Vector2d())
      .def("__mul__",
           [](const SO2Type& lhs, const Eigen::Matrix2d& rhs) { return lhs * SO2Type(rhs); })
      .def_static("hat", &SO2Type::hat)
      .def_static("vee", &SO2Type::vee)
      .def_static("exp", &SO2Type::exp);

  py::class_<SO3Type>(m, "SO3")
      .def(py::init<>())
      .def(py::init<const SO3Type&>())
      .def(py::init<const Eigen::Matrix3d&>())
      .def("adjoint", &SO3Type::Adj)
      .def("inverse", &SO3Type::inverse)
      .def("log", &SO3Type::log)
      .def("normalize", &SO3Type::normalize)
      .def("matrix", &SO3Type::matrix)
      .def("set_quaternion", &SO3Type::setQuaternion)
      .def("unit_quaternion", &SO3Type::unit_quaternion)
      .def("copy", [](const SO3Type& so3) { return SO3Type(so3); })
      .def(py::self * py::self)
      .def(py::self * Eigen::Vector3d())
      .def("__mul__",
           [](const SO3Type& lhs, const Eigen::Matrix3d& rhs) { return lhs * SO3Type(rhs); })
      .def_static("hat", &SO3Type::hat)
      .def_static("vee", &SO3Type::vee)
      .def_static("exp", &SO3Type::exp);

  py::class_<SE2Type>(m, "SE2")
      .def(py::init<>())
      .def(py::init<const SE2Type&>())
      .def(py::init<const Eigen::Matrix3d&>())
      .def(py::init<const Eigen::Matrix2d&, const Eigen::Vector2d&>())
      .def("adjoint", &SE2Type::Adj)
      .def("inverse", &SE2Type::inverse)
      .def("log", &SE2Type::log)
      .def("normalize", &SE2Type::normalize)
      .def("matrix", &SE2Type::matrix)
      .def("rotation_matrix", &SE2Type::rotationMatrix)
      .def("translation", py::overload_cast<>(&SE2Type::translation, py::const_))
      .def("so3", py::overload_cast<>(&SE2Type::so2, py::const_))
      .def("copy", [](const SE2Type& se2) { return SE2Type(se2); })
      .def(py::self * py::self)
      .def(py::self * Eigen::Vector2d())
      .def("__mul__",
           [](const SE2Type& lhs, const Eigen::Matrix3d& rhs) { return lhs * SE2Type(rhs); })
      .def_static("hat", &SE2Type::hat)
      .def_static("exp", &SE2Type::exp);

  py::class_<SE3Type>(m, "SE3")
      .def(py::init<>())
      .def(py::init<const SE3Type&>())
      .def(py::init<const Eigen::Matrix4d&>())
      .def(py::init<const Eigen::Matrix3d&, const Eigen::Vector3d&>())
      .def("adjoint", &SE3Type::Adj)
      .def("inverse", &SE3Type::inverse)
      .def("log", &SE3Type::log)
      .def("normalize", &SE3Type::normalize)
      .def("matrix", &SE3Type::matrix)
      .def("rotation_matrix", &SE3Type::rotationMatrix)
      .def("translation", py::overload_cast<>(&SE3Type::translation, py::const_))
      .def("so3", py::overload_cast<>(&SE3Type::so3, py::const_))
      .def("copy", [](const SE3Type& se3) { return SE3Type(se3); })
      .def(py::self * py::self)
      .def(py::self * Eigen::Vector3d())
      .def("__mul__",
           [](const SE3Type& lhs, const Eigen::Matrix4d& rhs) { return lhs * SE3Type(rhs); })
      .def_static("hat", &SE3Type::hat)
      .def_static("exp", &SE3Type::exp);
}
