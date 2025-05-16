from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from torch.utils.cpp_extension import CUDAExtension, CppExtension
import setuptools.command.build_ext
import os
import sys
from setuptools import Extension


# Create a modified BuildExtension class that works with Hatchling
class CustomBuildExtension(setuptools.command.build_ext.build_ext):
    def __init__(self):
        # Create a mock Distribution object
        from setuptools.dist import Distribution
        dist = Distribution()

        # Initialize with the mock distribution
        super().__init__(dist)


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        build_data["force_include"].update({
            "featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp": "featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp",
            "featup/adaptive_conv_cuda/adaptive_conv_kernel.cu": "featup/adaptive_conv_cuda/adaptive_conv_kernel.cu",
            "featup/adaptive_conv_cuda/adaptive_conv.cpp": "featup/adaptive_conv_cuda/adaptive_conv.cpp",
        })

        # Initialize the build extension with our custom class
        self.extension_builder = CustomBuildExtension()

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
        # Set up the extension builder with our extensions
        self.extension_builder.extensions = self.ext_modules

        # Get the current directory
        old_dir = os.getcwd()

        try:
            # Change to the directory containing the source files
            os.chdir(os.path.dirname(artifact_path))

            # Build the extensions
            self.extension_builder.build_extensions()

            # Copy the built extensions to the artifact path
            import shutil
            for ext in self.ext_modules:
                ext_path = self.extension_builder.get_ext_fullpath(ext.name)
                if os.path.exists(ext_path):
                    dest_path = os.path.join(artifact_path, os.path.basename(ext_path))
                    shutil.copy2(ext_path, dest_path)
        finally:
            # Restore the original directory
            os.chdir(old_dir)

        return super().finalize(version, build_data, artifact_path)
