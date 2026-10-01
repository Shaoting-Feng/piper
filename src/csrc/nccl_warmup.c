// src._nccl: Python binding for ncclWarmup (github.com/Shaoting-Feng/nccl).
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Declared here instead of including nccl.h, which needs the full CUDA toolkit headers.
typedef int ncclResult_t;
#define ncclSuccess 0
ncclResult_t ncclWarmup(int cudaDev);
const char* ncclGetErrorString(ncclResult_t result);

static PyObject* warmup(PyObject* self, PyObject* args) {
  int cuda_dev;
  if (!PyArg_ParseTuple(args, "i", &cuda_dev)) return NULL;
  ncclResult_t res;
  Py_BEGIN_ALLOW_THREADS
  res = ncclWarmup(cuda_dev);
  Py_END_ALLOW_THREADS
  if (res != ncclSuccess) {
    PyErr_Format(PyExc_RuntimeError, "ncclWarmup(%d) failed: %s", cuda_dev, ncclGetErrorString(res));
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
  {"warmup", warmup, METH_VARARGS,
   "warmup(cuda_dev): run NCCL's once-per-process setup for this device."},
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_nccl", NULL, -1, methods};

PyMODINIT_FUNC PyInit__nccl(void) { return PyModule_Create(&module); }
