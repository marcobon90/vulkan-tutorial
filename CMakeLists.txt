cmake_minimum_required(VERSION 3.29)
project(VulkanApp CXX)

# Imposta lo standard C++17 o C++20
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Trova Vulkan (Usa automaticamente $VULKAN_SDK se impostato,
# altrimenti la versione di sistema)
find_package(Vulkan REQUIRED)

# 2. Trova GLFW (Gestione finestra e input)
find_package(glfw3 REQUIRED)

# 3. GLM, TinyGLTF e TinyObjLoader sono header-only su Linux:
# Non servono moduli "Find", basta dire a CMake di cercare
# gli file header in /usr/include
include_directories(/usr/include)

# 4. Definisci l'eseguibile a partire dal sorgente
add_executable(VulkanApp main.cpp)

# 5. Collega le librerie compilate (Vulkan e GLFW) all'eseguibile
target_link_libraries(VulkanApp PRIVATE Vulkan::Vulkan glfw)
