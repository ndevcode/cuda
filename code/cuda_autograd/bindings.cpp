#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "memory_alloc.h"
#include "my_cuda_kernels.cuh"

namespace py = pybind11;

PYBIND11_MODULE(my_cuda_ops,m) {
    m.doc() = "Pybind11 example for custom CUDA kernels";

    m.def("relu_forward", [](py::array_t<float> input_array) {
        py::buffer_info buf = input_array.request();
        int sz = buf.size;

        MemoryManager mmgr;

        auto hd_in = mmgr.allocate_host_and_device<float>(sz);
        auto hd_out = mmgr.allocate_host_and_device<float>(sz);

        std::memcpy(hd_in.hostPtr,buf.ptr, sz * sizeof(float));

        hd_in.copy_to_device(sz);

        relu_forward(hd_in.devicePtr,hd_out.devicePtr,sz);

        hd_out.copy_to_host(sz);

        auto result = py::array_t<float>(sz);
        std::memcpy(result.mutable_data(), hd_out.hostPtr, sz * sizeof(float));
        result.resize(buf.shape);

        return result;
    });

    m.def("relu_backward", [](py::array_t<float> input_array,
                              py::array_t<float> grad_out_array) {
        py::buffer_info buf_in = input_array.request();
        py::buffer_info buf_grad_out = grad_out_array.request();
        int size = buf_in.size;

        MemoryManager mmgr;

        auto hd_in      = mmgr.allocate_host_and_device<float>(size);
        auto hd_gradOut = mmgr.allocate_host_and_device<float>(size);
        auto hd_gradIn  = mmgr.allocate_host_and_device<float>(size);

        std::memcpy(hd_in.hostPtr, buf_in.ptr, size * sizeof(float));
        std::memcpy(hd_gradOut.hostPtr, buf_grad_out.ptr, size * sizeof(float));

        hd_in.copy_to_device(size);
        hd_gradOut.copy_to_device(size);

        relu_backward(hd_in.devicePtr, hd_gradOut.devicePtr, hd_gradIn.devicePtr, size);

        hd_gradIn.copy_to_host(size);

        auto result = py::array_t<float>(size);
        std::memcpy(result.mutable_data(), hd_gradIn.hostPtr, size * sizeof(float));
        result.resize(buf_in.shape);

        return result;
    });
}
