cmake_minimum_required(VERSION 3.29)
project(CompShaderApp CXX)

# Imposta lo standard C++
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Trova Vulkan
find_package(Vulkan REQUIRED)

# 2. Trova GLFW
find_package(glfw3 REQUIRED)

# 3. Trova l'eseguibile slangc nel PATH di sistema o nel VULKAN_SDK
find_program(SLANGC_EXECUTABLE slangc REQUIRED)

# Include headers di sistema
include_directories(/usr/include)

# 4. Definisci l'eseguibile
add_executable(CompShaderApp main.cpp)

# 5. Linka le librerie
target_link_libraries(CompShaderApp PRIVATE Vulkan::Vulkan glfw)

# 6. Compilazione automatica degli Shader Slang
function(add_slang_shader_target TARGET_NAME)
    cmake_parse_arguments("SHADER" "" "" "SOURCES" ${ARGN})

    set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(OUTPUT_SPV ${SHADERS_DIR}/slang.spv)

    # Crea la cartella shaders se non esiste
    add_custom_command(
        OUTPUT ${SHADERS_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    )

    # Comando per compilare il file .slang in .spv
    add_custom_command(
        OUTPUT ${OUTPUT_SPV}
        COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES}
          -target spirv
          -profile spirv_1_4
          -emit-spirv-directly
          -fvk-use-entrypoint-name
          -entry vertMain
          -entry fragMain
          -entry compMain
          -o ${OUTPUT_SPV}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
        COMMENT "Compilazione dello shader Slang in SPIR-V..."
        VERBATIM
    )

    # Crea un target logico per la compilazione dello shader
    add_custom_target(${TARGET_NAME} DEPENDS ${OUTPUT_SPV})
endfunction()

# Definisci i file sorgente dello shader
set(MY_SLANG_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang)

# Crea il target per lo shader chiamato "CompileShaders"
add_slang_shader_target(CompileShaders SOURCES ${MY_SLANG_SOURCES})

# Dici a CMake che CompShaderApp DIPENDE da CompileShaders:
# In questo modo, quando fai la build di CompShaderApp,
#  lo shader verrà compilato automaticamente per primo!
add_dependencies(CompShaderApp CompileShaders)
