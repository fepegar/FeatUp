from hatchling.builders.hooks.plugin.interface import BuildHookInterface
import os
import sys
import importlib
import warnings
from setuptools import Extension

# Try to import CUDA extensions, but continue if not available
try:
    from torch.utils.cpp_extension import CUDAExtension, CppExtension
    import setuptools.command.build_ext
    CUDA_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch CUDA extensions not available - building without CUDA support")
    CUDAExtension = None
    CppExtension = None
    CUDA_AVAILABLE = False


# Create a modified BuildExtension class that works with Hatchling
if CUDA_AVAILABLE:
    class CustomBuildExtension(setuptools.command.build_ext.build_ext):
        def __init__(self):
            # Create a mock Distribution object
            from setuptools.dist import Distribution
            dist = Distribution()

            # Initialize with the mock distribution
            super().__init__(dist)

            # Set up build directories
            self.build_temp = os.path.abspath("build/temp")
            self.build_lib = os.path.abspath("build/lib")

            # Create necessary directories
            os.makedirs(self.build_temp, exist_ok=True)
            os.makedirs(self.build_lib, exist_ok=True)
else:
    # Dummy class when CUDA is not available
    class CustomBuildExtension:
        def __init__(self):
            self.extensions = []
            self.build_lib = "build/lib"

        def finalize_options(self):
            pass

        def run(self):
            warnings.warn("Skipping CUDA extensions - not available")

        def build_extensions(self):
            pass


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        print("Initializing custom build hook for CUDA extensions")

        # Make sure source files are included in the build
        build_data["force_include"].update({
            "featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp": "featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp",
            "featup/adaptive_conv_cuda/adaptive_conv_kernel.cu": "featup/adaptive_conv_cuda/adaptive_conv_kernel.cu",
            "featup/adaptive_conv_cuda/adaptive_conv.cpp": "featup/adaptive_conv_cuda/adaptive_conv.cpp",
        })

        try:
            # Initialize the build extension with our custom class
            self.extension_builder = CustomBuildExtension()

            # Define the extensions - use full paths for source files
            import os
            base_path = os.path.abspath(os.getcwd())

            self.ext_modules = [
                CUDAExtension(
                    'featup.adaptive_conv_cuda_impl',
                    [
                        os.path.join(base_path, 'featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp'),
                        os.path.join(base_path, 'featup/adaptive_conv_cuda/adaptive_conv_kernel.cu'),
                    ]
                ),
                CppExtension(
                    'featup.adaptive_conv_cpp_impl',
                    [os.path.join(base_path, 'featup/adaptive_conv_cuda/adaptive_conv.cpp')],
                    undef_macros=["NDEBUG"]
                ),
            ]

            print(f"Created {len(self.ext_modules)} extension modules")

        except Exception as e:
            print(f"Error setting up extension modules: {e}")
            import traceback
            traceback.print_exc()
            # Continue without extensions
            self.ext_modules = []

    def finalize(self, version, build_data, artifact_path):
        # Set up the extension builder with our extensions
        self.extension_builder.extensions = self.ext_modules

        # Configure the inplace option for editable installs
        self.extension_builder.inplace = 1

        # Get the current directory
        old_dir = os.getcwd()

        try:
            # Print what we're doing for debugging
            print(f"Building extensions at {artifact_path}")

            # Configure the extension builder
            self.extension_builder.finalize_options()

            # Build the extensions
            self.extension_builder.run()

            # Copy the built extensions to the artifact path
            import shutil
            import glob

            # Find all shared object files in the build directory
            build_dir = self.extension_builder.build_lib
            print(f"Looking for built extensions in {build_dir}")

            # Look for .so files that match our extension names
            for ext in self.ext_modules:
                pattern = os.path.join(build_dir, "**", f"{ext.name}*.so")
                for ext_file in glob.glob(pattern, recursive=True):
                    if os.path.exists(ext_file):
                        print(f"Found extension: {ext_file}")
                        target_dir = os.path.join(artifact_path, os.path.dirname(ext_file)[len(build_dir):].lstrip('/'))
                        os.makedirs(target_dir, exist_ok=True)
                        dest_path = os.path.join(target_dir, os.path.basename(ext_file))
                        print(f"Copying to {dest_path}")
                        shutil.copy2(ext_file, dest_path)
                    else:
                        print(f"Warning: Could not find built extension for {ext.name}")
        except Exception as e:
            print(f"Error during extension building: {e}")
            import traceback
            traceback.print_exc()
            # Continue even if extension building fails - the pure Python part will still work
        finally:
            # Restore the original directory
            os.chdir(old_dir)

        return super().finalize(version, build_data, artifact_path)
