#define PY_ARRAY_UNIQUE_SYMBOL cs2rl_binding_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

static PyMethodDef binding_methods[] = {{NULL, NULL, 0, NULL}};

static struct PyModuleDef binding_module = {
    PyModuleDef_HEAD_INIT, "binding", NULL, -1, binding_methods};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array1(NULL);
    return PyModule_Create(&binding_module);
}
