# Build the project

Now you need to install the Vulkan SDK:
1. Download the tarball from https://vulkan.lunarg.com/
2. Extract it to a convenient location, for example:
   mkdir -p ~/vulkansdk
   tar -xf vulkansdk-linux-x86_64-<version>.tar.xz -C ~/vulkansdk
   cd ~/vulkansdk
   ln -s <version> default

3. Add the following to your ~/.bashrc or ~/.zshrc:
   source ~/vulkansdk/default/setup-env.sh

4. Restart your terminal or run: source ~/.bashrc

5. Verify installation by running: vkcube

All dependencies have been installed successfully!
You can now use CMake to build your Vulkan project:
cmake -B build -S . -G Ninja
