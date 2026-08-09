#include "vulkan/vulkan.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_to_string.hpp>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
// #include <GLFW/glfw3native.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector<char const *> validation_layers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enable_validation_layers = false;
#else
constexpr bool enable_validation_layers = true;
#endif

class HelloTriangleApplication {
public:
  void run() {
    printf("Ho completato la parte sulla graphics pipeline basics!\n");
    printf("Ref: https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/02_Graphics_pipeline_basics/00_Introduction.html\n");
    init_window();
    init_vulkan();
    main_loop();
    cleanup();
  }

private:
  GLFWwindow *window = nullptr;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::DebugUtilsMessengerEXT debug_messenger = nullptr;
  vk::raii::PhysicalDevice physical_device = nullptr;
  vk::PhysicalDeviceFeatures physical_device_features;
  vk::raii::Device device = nullptr;
  vk::raii::Queue graphics_queue = nullptr;
  vk::raii::SwapchainKHR swap_chain = nullptr;
  std::vector<vk::Image> swap_chain_image;
  vk::SurfaceFormatKHR swap_chain_surface_format;
  vk::Extent2D swap_chain_extent;
  std::vector<vk::raii::ImageView> swap_chain_image_views;
  vk::raii::PipelineLayout pipeline_layout = nullptr;
  vk::raii::Pipeline graphics_pipeline = nullptr;

  std::vector<const char *> required_device_extension = {
      vk::KHRSwapchainExtensionName};

  void init_window() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vukan", nullptr, nullptr);
  }

  void init_vulkan() {
    createInstance();
    setup_debug_messenger();
    create_surface();
    pick_physical_device();
    create_logical_device();
    create_swap_chain();
    create_image_views();
    create_graphics_pipeline();
  }

  void main_loop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
    glfwDestroyWindow(window);

    glfwTerminate();
  }

  void createInstance() {
    vk::ApplicationInfo const appInfo =
        vk::ApplicationInfo()
            .setPApplicationName("Hello Triangle")
            .setPEngineName("No Engine")
            .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
            .setApiVersion(vk::ApiVersion14);

    // Get required layers
    std::vector<char const *> required_layers;
    if (enable_validation_layers) {
      required_layers.assign(validation_layers.begin(),
                             validation_layers.end());
    }

    auto layer_properties = context.enumerateInstanceLayerProperties();

    // Cerca in required_layers il PRIMO layer che NON viene trovato in
    // layer_properties
    auto unsupported_layer_it = std::find_if(
        required_layers.begin(), required_layers.end(),
        [&layer_properties](auto const &required_layer) {
          // Catturo layer_properties. required_layer è un singolo layer
          // richiesto. none_of restituisce true se NESSUN layer della GPU
          // corrisponde a required_layer.
          return std::none_of(layer_properties.begin(), layer_properties.end(),
                              [required_layer](auto const &layer_property) {
                                // Confronta il singolo layer GPU con il singolo
                                // layer richiesto
                                return strcmp(layer_property.layerName,
                                              required_layer) ==
                                       0; // == 0 FONDAMENTALE!
                              });
        });

    // Confronto la posizione dell'elemento trovato.
    //  Se risulta uguale a required_layers.end() cioè la posizione successiva
    //  all'ultimo elemento Allora tutti tutti i layer sono supportati.
    if (unsupported_layer_it != required_layers.end()) {
      throw std::runtime_error("Required layer not supported: " +
                               std::string(*unsupported_layer_it));
    }

    // Get  the required extensions
    auto required_exts = get_required_instance_extensions();
    auto extension_properties = context.enumerateInstanceExtensionProperties();

    auto unsupported_property_it = std::find_if(
        required_exts.begin(), required_exts.end(),
        [&extension_properties](auto const &required_extension) {
          return std::none_of(
              extension_properties.begin(), extension_properties.end(),
              [required_extension](auto const &extension_property) {
                return strcmp(extension_property.extensionName,
                              required_extension) == 0;
              });
        });

    if (unsupported_property_it != required_exts.end()) {
      throw std::runtime_error("Required extension not supported: " +
                               std::string(*unsupported_property_it));
    }

    vk::InstanceCreateInfo const createInfo =
        vk::InstanceCreateInfo()
            .setPApplicationInfo(&appInfo)
            .setEnabledLayerCount(static_cast<uint32_t>(required_layers.size()))
            .setPpEnabledLayerNames(required_layers.data())
            .setEnabledExtensionCount(
                static_cast<uint32_t>(required_exts.size()))
            .setPpEnabledExtensionNames(required_exts.data());

    instance = vk::raii::Instance(context, createInfo);
  }

  bool is_device_suitable(vk::raii::PhysicalDevice const &physical_device) {
    bool supported_vulkan1_3 = physical_device.getProperties().apiVersion;

    auto queue_families = physical_device.getQueueFamilyProperties();
    bool supports_graphics =
        std::any_of(queue_families.begin(), queue_families.end(),
                    [](vk::QueueFamilyProperties const &qfp) {
                      return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
                    });
    auto available_device_extensions =
        physical_device.enumerateDeviceExtensionProperties();
    bool supported_all_required_extensions = std::all_of(
        required_device_extension.begin(), required_device_extension.end(),
        [&available_device_extensions](auto const &required_device_extension) {
          return std::any_of(
              available_device_extensions.begin(),
              available_device_extensions.end(),
              [required_device_extension](
                  auto const &available_device_extension) {
                return strcmp(available_device_extension.extensionName,
                              required_device_extension) == 0;
              });
        });

    auto features = physical_device.getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

    bool supports_required_features =
        features.get<vk::PhysicalDeviceVulkan11Features>()
            .shaderDrawParameters &&
        features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
        features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState;

    return supports_required_features && supported_all_required_extensions &&
           supported_vulkan1_3 && supports_graphics;
  }

  void create_surface() {
    VkSurfaceKHR raw_surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &raw_surface) ==
        0) {
      surface = vk::raii::SurfaceKHR(instance, raw_surface);
    }
    throw std::runtime_error("failed to create window surface!");
  }
  void pick_physical_device() {
    std::vector<vk::raii::PhysicalDevice> physical_devices =
        instance.enumeratePhysicalDevices();
    auto const dev_iter =
        std::find_if(physical_devices.begin(), physical_devices.end(),
                     [&](vk::raii::PhysicalDevice const &pd) {
                       return is_device_suitable(pd);
                     });
    if (dev_iter == physical_devices.end())
      throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    physical_device = *dev_iter;
  }

  void create_logical_device() {
    std::vector<vk::QueueFamilyProperties> queue_family_properties =
        physical_device.getQueueFamilyProperties();

    uint32_t queue_index = ~0;

    for (uint32_t qfp_index = 0; qfp_index < queue_family_properties.size();
         qfp_index++) {
      if ((queue_family_properties[qfp_index].queueFlags &
           vk::QueueFlagBits::eGraphics) &&
          physical_device.getSurfaceSupportKHR(qfp_index, *surface)) {
        queue_index = qfp_index;
        break;
      }
    }

    if (queue_index == ~0)
      throw std::runtime_error(
          "Could not find a queue for graphics and present -> terminating");

    auto feature_chain =
        vk::StructureChain<vk::PhysicalDeviceFeatures2,
                           vk::PhysicalDeviceVulkan11Features,
                           vk::PhysicalDeviceVulkan13Features,
                           vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>{
            vk::PhysicalDeviceFeatures2{},
            vk::PhysicalDeviceVulkan11Features().setShaderDrawParameters(true),
            vk::PhysicalDeviceVulkan13Features().setDynamicRendering(true),
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT()
                .setExtendedDynamicState(true)};

    float queue_priority = 0.5f;
    auto device_queue_create_info = vk::DeviceQueueCreateInfo()
                                        .setQueueFamilyIndex(queue_index)
                                        .setQueueCount(1)
                                        .setPQueuePriorities(&queue_priority);

    auto device_create_info =
        vk::DeviceCreateInfo()
            .setPNext(&feature_chain.get<vk::PhysicalDeviceFeatures2>())
            .setQueueCreateInfoCount(1)
            .setPQueueCreateInfos(&device_queue_create_info)
            .setEnabledExtensionCount(
                static_cast<uint32_t>(required_device_extension.size()))
            .setPpEnabledExtensionNames(required_device_extension.data());

    device = vk::raii::Device(physical_device, device_create_info);
    graphics_queue = vk::raii::Queue(device, queue_index, 0);
  }

  void setup_debug_messenger() {
    if (!enable_validation_layers)
      return;

    vk::DebugUtilsMessageSeverityFlagsEXT severity_flags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

    vk::DebugUtilsMessageTypeFlagsEXT message_type_flags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

    vk::DebugUtilsMessengerCreateInfoEXT const debug_utils_create_info_ext =
        vk::DebugUtilsMessengerCreateInfoEXT()
            .setMessageSeverity(severity_flags)
            .setMessageType(message_type_flags)
            .setPfnUserCallback(&debug_callback);

    debug_messenger =
        instance.createDebugUtilsMessengerEXT(debug_utils_create_info_ext);
  }
  const std::vector<const char *> get_required_instance_extensions() {
    uint32_t glfw_ext_count = 0;
    auto glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    std::vector extensions(glfw_exts, glfw_exts + glfw_ext_count);
    if (enable_validation_layers) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    return extensions;
  }

  void create_swap_chain() {
    vk::SurfaceCapabilitiesKHR surface_capabilities =
        physical_device.getSurfaceCapabilitiesKHR(*surface);
    swap_chain_extent = choose_swap_extent(surface_capabilities);
    uint32_t min_image_count =
        choose_swap_min_image_count(surface_capabilities);

    std::vector<vk::SurfaceFormatKHR> available_formats =
        physical_device.getSurfaceFormatsKHR(*surface);
    swap_chain_surface_format = choose_swap_surface_format(available_formats);

    std::vector<vk::PresentModeKHR> available_present_modes =
        physical_device.getSurfacePresentModesKHR();
    vk::PresentModeKHR present_mode =
        choose_swap_present_mode(available_present_modes);

    auto swap_chain_create_info =
        vk::SwapchainCreateInfoKHR()
            .setSurface(*surface)
            .setMinImageCount(min_image_count)
            .setImageFormat(swap_chain_surface_format.format)
            .setImageColorSpace(swap_chain_surface_format.colorSpace)
            .setImageExtent(swap_chain_extent)
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
            .setImageSharingMode(vk::SharingMode::eExclusive)
            .setPreTransform(surface_capabilities.currentTransform)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
            .setPresentMode(present_mode)
            .setClipped(true);

    swap_chain = vk::raii::SwapchainKHR(device, swap_chain_create_info);
    swap_chain_image = swap_chain.getImages();
  }

  void create_image_views() {
    assert(swap_chain_image_views.empty());

    auto image_view_create_info =
        vk::ImageViewCreateInfo()
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(swap_chain_surface_format.format)
            .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    for (auto &image : swap_chain_image) {
      image_view_create_info.image = image;
      swap_chain_image_views.emplace_back(device, image_view_create_info);
    }
  }

  void create_graphics_pipeline() {
    auto shader_module = create_shader_module(read_file("shaders/shader.spv"));
    auto vet_shader_stage_info = vk::PipelineShaderStageCreateInfo()
                                     .setStage(vk::ShaderStageFlagBits::eVertex)
                                     .setModule(shader_module)
                                     .setPName("vertMain");
    auto vet_fragment_stage_info =
        vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(shader_module)
            .setPName("fragMain");

    vk::PipelineShaderStageCreateInfo shader_stages[] = {
        vet_shader_stage_info, vet_fragment_stage_info};
    vk::PipelineVertexInputStateCreateInfo vertex_input_info;
    auto input_assembly =
        vk::PipelineInputAssemblyStateCreateInfo().setTopology(
            vk::PrimitiveTopology::eTriangleList);

    vk::Viewport viewport{0.0f,
                          0.0f,
                          static_cast<float>(swap_chain_extent.width),
                          static_cast<float>(swap_chain_extent.height),
                          0.0f,
                          1.0f};

    auto scissor =
        vk::Rect2D().setOffset(vk::Offset2D(0, 0)).setExtent(swap_chain_extent);

    std::vector<vk::DynamicState> dynamic_states = {vk::DynamicState::eViewport,
                                                    vk::DynamicState::eScissor};

    auto dynamic_state =
        vk::PipelineDynamicStateCreateInfo()
            .setDynamicStateCount(static_cast<uint32_t>(dynamic_states.size()))
            .setPDynamicStates(dynamic_states.data());

    auto view_port_state = vk::PipelineViewportStateCreateInfo()
                               .setViewportCount(1)
                               .setScissorCount(1);

    auto rasterizer = vk::PipelineRasterizationStateCreateInfo()
                          .setDepthClampEnable(vk::False)
                          .setRasterizerDiscardEnable(vk::False)
                          .setPolygonMode(vk::PolygonMode::eFill)
                          .setCullMode(vk::CullModeFlagBits::eBack)
                          .setFrontFace(vk::FrontFace::eClockwise)
                          .setDepthBiasEnable(vk::False)
                          .setLineWidth(1.0f);

    auto multisampling =
        vk::PipelineMultisampleStateCreateInfo()
            .setRasterizationSamples(vk::SampleCountFlagBits::e1)
            .setSampleShadingEnable(vk::False);

    auto color_blend_attachment =
        vk::PipelineColorBlendAttachmentState()
            .setBlendEnable(vk::False)
            .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
            .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
            .setColorBlendOp(vk::BlendOp::eAdd)
            .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
            .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
            .setAlphaBlendOp(vk::BlendOp::eAdd)
            .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                               vk::ColorComponentFlagBits::eG |
                               vk::ColorComponentFlagBits::eB |
                               vk::ColorComponentFlagBits::eA);

    auto color_blending = vk::PipelineColorBlendStateCreateInfo()
                              .setLogicOpEnable(vk::False)
                              .setLogicOp(vk::LogicOp::eCopy)
                              .setAttachmentCount(1)
                              .setPAttachments(&color_blend_attachment);

    auto pipeline_layout_info = vk::PipelineLayoutCreateInfo();
    pipeline_layout_info.setLayoutCount = 0;
    pipeline_layout_info.pushConstantRangeCount = 0;

    pipeline_layout = vk::raii::PipelineLayout(device, pipeline_layout_info);

    auto pipeline_rendering_create_info =
        vk::PipelineRenderingCreateInfo()
            .setColorAttachmentCount(1)
            .setPColorAttachmentFormats(&swap_chain_surface_format.format);

    auto pipeline_create_info_chain =
        vk::StructureChain<vk::GraphicsPipelineCreateInfo,
                           vk::PipelineRenderingCreateInfo>{
            vk::GraphicsPipelineCreateInfo()
                .setStageCount(2)
                .setPStages(shader_stages)
                .setPVertexInputState(&vertex_input_info)
                .setPInputAssemblyState(&input_assembly)
                .setPViewportState(&view_port_state)
                .setPRasterizationState(&rasterizer)
                .setPMultisampleState(&multisampling)
                .setPColorBlendState(&color_blending)
                .setPDynamicState(&dynamic_state)
                .setLayout(pipeline_layout)
                .setRenderPass(nullptr),

            vk::PipelineRenderingCreateInfo()
                .setColorAttachmentCount(1)
                .setPColorAttachmentFormats(&swap_chain_surface_format.format)};

    graphics_pipeline = vk::raii::Pipeline(
        device, nullptr,
        pipeline_create_info_chain.get<vk::GraphicsPipelineCreateInfo>());
  }

  [[nodiscard]] vk::raii::ShaderModule
  create_shader_module(const std::vector<char> &code) const {
    auto create_info =
        vk::ShaderModuleCreateInfo()
            .setCodeSize(code.size() * sizeof(char))
            .setPCode(reinterpret_cast<const uint32_t *>(code.data()));

    auto shader_module = vk::raii::ShaderModule{device, create_info};

    return shader_module;
  }

  uint32_t choose_swap_min_image_count(
      vk::SurfaceCapabilitiesKHR const &surface_capabilities) {
    auto min_image_count = std::max(3u, surface_capabilities.minImageCount);
    if ((0 < surface_capabilities.maxImageCount) &&
        (surface_capabilities.maxImageCount < min_image_count))
      min_image_count = surface_capabilities.maxImageCount;
    return min_image_count;
  }
  vk::SurfaceFormatKHR choose_swap_surface_format(
      const std::vector<vk::SurfaceFormatKHR> &available_formats) {
    const auto format_it = std::find_if(
        available_formats.begin(), available_formats.end(),
        [](const vk::SurfaceFormatKHR &format) {
          return format.format == vk::Format::eR8G8B8A8Srgb &&
                 format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });
    return format_it != available_formats.end() ? *format_it
                                                : available_formats[0];
  }

  vk::PresentModeKHR choose_swap_present_mode(
      std::vector<vk::PresentModeKHR> const &available_present_mode) {
    assert(std::any_of(available_present_mode.begin(),
                       available_present_mode.end(),
                       [](vk::PresentModeKHR present_mode) {
                         return present_mode == vk::PresentModeKHR::eFifo;
                       }));
    return std::any_of(available_present_mode.begin(),
                       available_present_mode.end(),
                       [](const vk::PresentModeKHR value) {
                         return vk::PresentModeKHR::eMailbox == value;
                       })
               ? vk::PresentModeKHR::eMailbox
               : vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D
  choose_swap_extent(vk::SurfaceCapabilitiesKHR const &capabilities) {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    return {
        std::clamp<uint32_t>(width, capabilities.minImageExtent.width,
                             capabilities.maxImageExtent.width),
        std::clamp<uint32_t>(width, capabilities.minImageExtent.height,
                             capabilities.maxImageExtent.height),
    };
  }

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL
  debug_callback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                 vk::DebugUtilsMessageTypeFlagsEXT type,
                 const vk::DebugUtilsMessengerCallbackDataEXT *pCall_back_data,
                 void *pUser_data) {
    std::cerr << "validation layer: type " << vk::to_string(type)
              << " msg: " << pCall_back_data->pMessage << std::endl;
    return vk::False;
  }

  static std::vector<char> read_file(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file");
    }

    const auto file_size = static_cast<uint32_t>(file.tellg());
    std::vector<char> buffer(file_size);
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();
    return buffer;
  }
};

int main() {
  try {
    HelloTriangleApplication app;
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
