from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        build_data["force_include"].update({
            "featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp": "featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp",
            "featup/adaptive_conv_cuda/adaptive_conv_kernel.cu": "featup/adaptive_conv_cuda/adaptive_conv_kernel.cu",
            "featup/adaptive_conv_cuda/adaptive_conv.cpp": "featup/adaptive_conv_cuda/adaptive_conv.cpp",
        })

        # Initialize the build extension
        self.extension_builder = BuildExtension()

        # Define the extensions
        self.ext_modules = [
            CUDAExtension(
                'adaptive_conv_cuda_impl',
                [
                    'featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp',
                    'featup/adaptive_conv_cuda/adaptive_conv_kernel.cu',
                ]
            ),
            CppExtension(
                'adaptive_conv_cpp_impl',
                ['featup/adaptive_conv_cuda/adaptive_conv.cpp'],
                undef_macros=["NDEBUG"]
            ),
        ]

    def finalize(self, version, build_data, artifact_path):
        # Build the extensions
        self.extension_builder.build_extensions(self.ext_modules)
        return super().finalize(version, build_data, artifact_path)
