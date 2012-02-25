# Copyright (c) 2011 J. David Lee. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are
# met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

import os
import fcntl
import imp
import numpy as np


###############################################################################
# C-code skeleton.                                                            #
###############################################################################
_SKEL = r'''
#define PY_ARRAY_UNIQUE_SYMBOL __UNIQUE_NAME__
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

// Forward declarations of our function.
static PyObject *function(PyObject *self, PyObject *args); 


// Boilerplate: function list.
static PyMethodDef methods[] = {
    { "function", function, METH_VARARGS, "Doc string."},
    { NULL, NULL, 0, NULL } /* Sentinel */
};

// Boilerplate: Module initialization.
PyMODINIT_FUNC init__MODULE_NAME__(void) {
        (void) Py_InitModule("__MODULE_NAME__", methods);
        import_array();
}


/*****************************************************************************
 * Array access macros.                                                      *
 *****************************************************************************/
__NUMPY_ARRAY_MACROS__


/*****************************************************************************
 * Debug macros.                                                             *
 *****************************************************************************/
__NUMPY_ARRAY_DEBUG_MACROS__


/*****************************************************************************
 * Support code.                                                             *
 *****************************************************************************/
__SUPPORT_CODE__


/*****************************************************************************
 * The function.                                                             *
 *****************************************************************************/
static PyObject *function(PyObject *self, PyObject *args) {


/***************************************
 * Variable declarations.              *
 ***************************************/
__FUNC_VAR_DECLARATIONS__


/***************************************
 * Parse variables.                    *
 ***************************************/
if (!PyArg_ParseTuple(args, "__PARSE_ARG_TYPES__", __PARSE_ARG_LIST__)) {
    return NULL;
} 


/***************************************
 * User code.                          *
 ***************************************/
__USER_CODE__


/***************************************
 * Return value.                       *
 ***************************************/
__RETURN_VAL__

} // End of function(self, args).

'''

###############################################################################
# Module setup.                                                               #
###############################################################################
_PATH = os.path.expanduser('~/.np_inline')

try:
    os.makedirs(_PATH)
except:
    pass


###############################################################################
# Code generation.                                                            #
###############################################################################
_TYPE_CONV_DICT = {
    float : 'double',
    int   : 'long'
}


_RETURN_FUNC_DICT = {
    float : 'PyFloat_FromDouble',
    int   : 'PyLong_FromLong'
}


_TYPE_PARSE_SPEC_DICT = {
    float : 'd',
    int   : 'l'
}


_NP_TYPE_CONV_DICT = {
    np.uint8    : 'npy_uint8',
    np.uint16   : 'npy_uint16',
    np.uint32   : 'npy_uint32',
    np.uint64   : 'npy_uint64',
    np.int8     : 'npy_int8',
    np.int16    : 'npy_int16',
    np.int32    : 'npy_int32',
    np.int64    : 'npy_int64',
    np.float32  : 'npy_float32',
    np.float64  : 'npy_float64',
}

# The numpy floating point 128 bit type isn't available on all systems it 
# turns out. 
try:
    _NP_TYPE_CONV_DICT[np.float128] = 'npy_float128'
except:
    print('Numpy float128 not available.')


###############################################################################
# Code generation functions.
###############################################################################
def _gen_var_decls(py_types, np_types, return_type):
    """py_types should be a list with elements of the form
    (python_object, python_type, c_code_name)
    
    np_types should be a list with elements of the form 
    (numpy_object, numpy_type, num_dims, c_code_name)
    """
    str_list = []
    for py_type, c_name in py_types:
        c_type = _TYPE_CONV_DICT[py_type]
        str_list.append('{0} {1};'.format(c_type, c_name))

    for np_type, dims, c_name in np_types:
        str_list.append('PyArrayObject *py_{0};'.format(c_name))

    if return_type is not None:
        c_type = _TYPE_CONV_DICT[return_type]
        str_list.append('{0} return_val;'.format(c_type))

    return '\n'.join(str_list)


def _gen_parse_arg_types(py_types, np_types):
    str_list = []
    for py_type, c_name in py_types:
        str_list.append(_TYPE_PARSE_SPEC_DICT[py_type])

    for np_type, dims, c_name in np_types:
        str_list.append('O!')
        
    return ''.join(str_list)


def _gen_parse_arg_list(py_types, np_types):
    str_list = []

    for py_type, c_name in py_types:
        str_list.append('&{0}'.format(c_name))

    for np_type, dims, c_name in np_types:
        str_list.append('&PyArray_Type, &py_{0}'.format(c_name))
        
    return ', '.join(str_list)


def _gen_numpy_array_index_macro(np_type, dims, c_name):
    c_type = _NP_TYPE_CONV_DICT[np_type]

    # First we generate the list of arguments for the macro. 
    # These have the form x0, x1, ...
    arg_list = ', '.join(['x{0}'.format(i) for i in range(dims)])

    # Next, we want to create the indexing code. This looks like:
    # *(type *)((data + i*array->strides[0] + j*array->strides[1]))
    strides = ''
    for i in range(dims):
        strides += ' + (x{0}) * PyArray_STRIDES(py_{1})[{0}]'.format(i, c_name)
    
    return '#define {0}({1}) (*({2} *)((PyArray_BYTES(py_{0}) {3})))'.format(
        c_name, arg_list, c_type, strides)


def _gen_numpy_array_macros(np_types):
    str_list = []
    for np_type, dims, c_name in np_types:
        str_list.append(_gen_numpy_array_index_macro(np_type, dims, c_name))
        s = '#define {0}_shape(i) (py_{0}->dimensions[(i)])'.format(c_name)
        str_list.append(s)
        s = '#define {0}_ndim (py_arr->nd)'.format(c_name)
        str_list.append(s)
    return '\n'.join(str_list)


def _gen_numpy_array_debug_macros(np_types):
    str_list = []
    for np_type, dims, c_name in np_types:
        arg_list = ','.join(['x{0}'.format(i) for i in range(dims)])
        
        str_list.append('#define {0}_assert({1}) \\'.format(c_name, arg_list))
        for i in range(dims):
            str_list.append('assert({0} < py_{1}->nd);\\'.format(i, c_name))
            str_list.append('assert((x{0}) >= 0);\\'.format(i))
            str_list.append(
                'assert((x{0}) < py_{1}->dimensions[({0})]) \\'.format(
                    i, c_name))
    return '\n'.join(str_list)[:-1]

            

def _gen_return_val(return_type):
    if return_type is None:
        return 'Py_RETURN_NONE;'
    return 'return {0}(return_val);'.format(_RETURN_FUNC_DICT[return_type])


def _gen_code(name, user_code, py_types, np_types, support_code, return_type):
    """Return a string containing the generated C code."""
    s = _SKEL.replace('__MODULE_NAME__', 
                     name)
    s = s.replace('__UNIQUE_NAME__', 
                  '__np_inline_{0}'.format(name))
    s = s.replace('__NUMPY_ARRAY_MACROS__', 
                  _gen_numpy_array_macros(np_types))
    s = s.replace('__NUMPY_ARRAY_DEBUG_MACROS__',
                  _gen_numpy_array_debug_macros(np_types))
    s = s.replace('__SUPPORT_CODE__', 
                  support_code)
    s = s.replace('__FUNC_VAR_DECLARATIONS__', 
                  _gen_var_decls(py_types, np_types, return_type))
    s = s.replace('__PARSE_ARG_TYPES__',
                  _gen_parse_arg_types(py_types, np_types))
    s = s.replace('__PARSE_ARG_LIST__', 
                  _gen_parse_arg_list(py_types, np_types))
    s = s.replace('__USER_CODE__', 
                  user_code)
    s = s.replace('__RETURN_VAL__', 
                  _gen_return_val(return_type))
    return s


###############################################################################
# Building and installation.                                                  #
###############################################################################
def _build_install_module(code_str, mod_name, extension_kwargs):
    # Save the current path so we can reset at the end of this function.
    curpath = os.getcwd() 
    mod_name_c = '{0}.c'.format(mod_name)

    # Change to the code directory.
    os.chdir(_PATH)

    try:
        from distutils.core import setup, Extension

        with open(mod_name_c, 'w') as f:
            # Write out the code.
            f.write(code_str)
                
        # Make sure numpy headers are included. 
        if 'include_dirs' not in extension_kwargs:
            extension_kwargs['include_dirs'] = []
        extension_kwargs['include_dirs'].append(np.get_include())
            
        # Create the extension module object. 
        ext = Extension(mod_name, [mod_name_c], **extension_kwargs)
            
        # Clean.
        setup(ext_modules=[ext], script_args=['clean'])
            
        # Build and install the module here. 
        setup(ext_modules=[ext], 
              script_args=['install', '--install-lib={0}'.format(_PATH)])
            
    finally:
        os.chdir(curpath)


###############################################################################
# File locking: This works on linux, but may not work on other platforms. 
###############################################################################
def _get_lock(mod_name):
    lock_file = os.path.join(_PATH, '{0}.lock'.format(mod_name))
    lock = open(lock_file, 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock
    

def _release_lock(lock):
    lock.close()


###############################################################################
# Helper functions.                                                           #
###############################################################################
def _string_or_path(code_str, code_path):
    """Return code_str if it is not None, or the contents in code_path.
    """
    if code_str is not None:
        return code_str
    
    if code_path is not None:
        with open(code_path, 'rb') as f:
            return f.read()

    return ''


###############################################################################
# The inline function itself.
###############################################################################
def inline(py_types=(), np_types=(), code=None, code_path=None, 
           support_code=None, support_code_path=None, 
           extension_kwargs={}, return_type=None, debug=False):
    """Inline C code in your python code. 
    
    Parameters:
    py_types : typle of tuples (python_type, c_name)
        Type specifications for non-numpy arguments. Currently only int 
        and float are valid types. Default is empty tuple.
    np_types : typle of tuples (numpy_type, dims, c_name)
        Type specifications for numpy-type arguments. Most numeric numpy 
        types are valid. dims is the integer number of dimensions of the 
        corresponding array. Default is empty tuple.
    code : string, optional 
        C-code. One of code and code_path must be given. 
    code_path : string, optional
        Full path to c-code. One of code or code_path should be given.
    support_code : string, optional 
        C support code. This code will be inserted before the function 
        containing the c-code above. This can include any valid C code 
        including #includes and #defines.
    support_code_path : string, optional
        Full path to support code. 
    extension_kwargs : dictionary, optiona
        Keyword arguments to pass to the distutils.Extension constructor.
    return_type : python primitive type
        Either int or float.
    debug : boolean
        If True, build with debugging enabled. 
    """
    # Get actual code strings. 
    code         = _string_or_path(code,         code_path)
    support_code = _string_or_path(support_code, support_code_path)
    
    # Make sure types are in tuples so they are hashable. 
    py_types = tuple(py_types)
    np_types = tuple(np_types)

    # Generate the module name. 
    h = abs(hash((str(py_types), str(np_types), code, support_code, debug)))
    mod_name = 'mod_{}'.format(h)

    # The module path. 
    mod_path = os.path.join(_PATH, '{0}.so'.format(mod_name))
    
    # Try to import the module directly. 
    try:
        mod = imp.load_dynamic(mod_name, mod_path)
        return mod.function
    except:
        pass

    # We'll have to compile the module. First get an exclusive lock. 
    lock = _get_lock(mod_name)

    try:
        # Generate the code string.
        code_str = _gen_code(mod_name, code, py_types, np_types, support_code,
                             return_type)

        # Modify keyword arguments for debugging. 
        if debug:
            if 'undef_macros' not in extension_kwargs:
                extension_kwargs['undef_macros'] = []
            if 'NDEBUG' not in extension_kwargs['undef_macros']:
                extension_kwargs['undef_macros'].append('NDEBUG')

        # Build the module. 
        _build_install_module(code_str, mod_name, extension_kwargs, )
        
        # Return the module. 
        mod = imp.load_dynamic(mod_name, mod_path)
        return mod.function

    finally:
        _release_lock(lock)
                              




