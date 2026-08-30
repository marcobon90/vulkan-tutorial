#include "vulkan/vulkan.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <ios>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_to_string.hpp>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint32_t PARTICLE_COUNT = 8192;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validation_layers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enable_validation_layers = false;
#else
constexpr bool enable_validation_layers = true;
#endif

/*
 * Using only delta time.
 */
struct UniformBufferObject {
  float delta_time = 1.0f;
};

struct Particle {
  glm::vec2 pos;
  glm::vec2 vel;
  glm::vec4 color;

  static vk::VertexInputBindingDescription get_binding_description() {
    return vk::VertexInputBindingDescription()
        .setBinding(0)
        .setStride(sizeof(Particle))
        .setInputRate(vk::VertexInputRate::eVertex);
  }

  static std::array<vk::VertexInputAttributeDescription, 2>
  get_attribute_descriptio() {
    return {
        vk::VertexInputAttributeDescription()
            .setLocation(0)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32Sfloat)
            .setOffset(offsetof(Particle, pos)),
        vk::VertexInputAttributeDescription()
            .setLocation(1)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32B32A32Sfloat)
            .setOffset(offsetof(Particle, color)),
    };
  }
};

class ComputeShaderApplication {
public:
  void run() {
    init_window();
    init_vulkan();
    main_loop();
    cleanup();
  }

private:
  GLFWwindow *window = nullptr;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debug_messenger = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;

  vk::raii::PhysicalDevice physical_device = nullptr;
  vk::raii::Device device = nullptr;
  uint32_t queue_index = ~0;
  vk::raii::Queue queue = nullptr;

  vk::raii::SwapchainKHR swap_chain = nullptr;
  std::vector<vk::Image> swap_chain_images;
  vk::SurfaceFormatKHR swap_chain_surface_format;
  vk::Extent2D swap_chain_extent;
  std::vector<vk::raii::ImageView> swap_chain_image_views;

  vk::raii::PipelineLayout pipeline_layout = nullptr;
  vk::raii::Pipeline graphics_pipeline = nullptr;

  vk::raii::DescriptorSetLayout compute_descriptor_set_layout = nullptr;
  vk::raii::PipelineLayout compute_pipeline_layout = nullptr;
  vk::raii::Pipeline compute_pipeline = nullptr;

  std::vector<vk::raii::Buffer> shader_storage_buffers;
  std::vector<vk::raii::DeviceMemory> shared_storage_buffers_memory;

  std::vector<vk::raii::Buffer> uniform_buffers;
  std::vector<vk::raii::DeviceMemory> uniform_buffers_memory;
  std::vector<void *> uniform_buffers_mapped;

  vk::raii::DescriptorPool descriptor_pool = nullptr;
  std::vector<vk::raii::DescriptorSet> compute_descriptor_sets;

  vk::raii::CommandPool command_pool = nullptr;
  std::vector<vk::raii::CommandBuffer> command_buffers;
  std::vector<vk::raii::CommandBuffer> compute_command_buffers;

  vk::raii::Semaphore semaphore = nullptr;
  uint64_t time_line_value = 0;
  std::vector<vk::raii::Fence> in_flight_fences;
  uint32_t frame_index = 0;

  double last_frame_time = 0.0;

  bool framebuffer_resized = false;

  double last_time = 0.0f;

  std::vector<const char *> required_device_extension = {
      vk::KHRSwapchainExtensionName};

  void init_window() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vukan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);

    last_time = glfwGetTime();
  }

  static void framebuffer_resize_callback(GLFWwindow *window, int width,
                                          int height) {
    auto app = reinterpret_cast<ComputeShaderApplication *>(
        glfwGetWindowUserPointer(window));
    app->framebuffer_resized = true;
  }

  void init_vulkan() {
    create_instance();
    setup_debug_messenger();
    create_surface();
    pick_physical_device();
    create_logical_device();
    create_swap_chain();
    create_image_views();
    create_compute_descriptor_set_layout();
    create_graphics_pipeline();
    create_compute_pipeline();
    create_command_pool();
    create_shader_storage_buffers();
    create_uniform_buffers();
    create_descriptor_pool();
    create_compute_descriptor_sets();
    create_command_buffers();
    create_compute_command_buffers();
    create_sync_objects();
  }

  void main_loop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      draw_frame();

      // Animate
      double current_time = glfwGetTime();
      last_frame_time = (current_time - last_time) * 1000.0;
      last_time = current_time;
    }

    device.waitIdle();
  }

  void cleanup_swapchain() {
    swap_chain.clear();
    swap_chain = nullptr;
  }

  void cleanup() {
    glfwDestroyWindow(window);

    glfwTerminate();
  }

  void recreate_swap_chain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }
    device.waitIdle();

    cleanup_swapchain();
    create_swap_chain();
    create_image_views();
  }

  void create_instance() {
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

  void create_surface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
      throw std::runtime_error("failed to create window surface!");
    }
    std::cout << "Surface creata" << std::endl;
    surface = vk::raii::SurfaceKHR(instance, _surface);
  }

  bool is_device_suitable(vk::raii::PhysicalDevice const &physical_device) {
    bool supported_vulkan1_3 =
        physical_device.getProperties().apiVersion >= VK_API_VERSION_1_3;

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
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>();

    bool supports_required_features =
        features.get<vk::PhysicalDeviceFeatures2>()
            .features.samplerAnisotropy &&
        features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
        features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState &&
        features.get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>()
            .timelineSemaphore;

    return supports_required_features && supported_all_required_extensions &&
           supported_vulkan1_3 && supports_graphics;
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

    auto feature_chain = vk::StructureChain<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>{
        vk::PhysicalDeviceFeatures2().setFeatures(
            vk::PhysicalDeviceFeatures().setSamplerAnisotropy(true)),
        vk::PhysicalDeviceVulkan13Features()
            .setSynchronization2(true)
            .setDynamicRendering(true),
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT()
            .setExtendedDynamicState(true),
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR().setTimelineSemaphore(
            true)};

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
    queue = vk::raii::Queue(device, queue_index, 0);
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
        physical_device.getSurfacePresentModesKHR(*surface);
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
    swap_chain_images = swap_chain.getImages();
  }

  void create_image_views() {
    assert(swap_chain_image_views.empty());

    auto image_view_create_info =
        vk::ImageViewCreateInfo()
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(swap_chain_surface_format.format)
            .setSubresourceRange(
                vk::ImageSubresourceRange()
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)
                    .setBaseMipLevel(0)
                    .setLevelCount(1)
                    .setBaseArrayLayer(0)
                    .setLayerCount(1));

    for (auto &image : swap_chain_images) {
      image_view_create_info.setImage(image);
      swap_chain_image_views.emplace_back(device, image_view_create_info);
    }
  }

  void create_compute_descriptor_set_layout() {
    std::array<vk::DescriptorSetLayoutBinding, 3> layout_bindings{
        vk::DescriptorSetLayoutBinding()
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eUniformBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute),
        vk::DescriptorSetLayoutBinding()
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute),
        vk::DescriptorSetLayoutBinding()
            .setBinding(2)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute)};

    auto layout_info =
        vk::DescriptorSetLayoutCreateInfo()
            .setBindingCount(static_cast<uint32_t>(layout_bindings.size()))
            .setPBindings(layout_bindings.data());

    compute_descriptor_set_layout =
        vk::raii::DescriptorSetLayout(device, layout_info);
  }

  void create_graphics_pipeline() {
    auto shader_module = create_shader_module(read_file("shaders/slang.spv"));
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

    auto binding_description = Particle::get_binding_description();
    auto attribute_description = Particle::get_attribute_descriptio();

    auto vertex_input_info =
        vk::PipelineVertexInputStateCreateInfo()
            .setVertexBindingDescriptionCount(1)
            .setPVertexBindingDescriptions(&binding_description)
            .setVertexAttributeDescriptionCount(
                static_cast<uint32_t>(attribute_description.size()))
            .setPVertexAttributeDescriptions(attribute_description.data());

    auto input_assembly = vk::PipelineInputAssemblyStateCreateInfo()
                              .setTopology(vk::PrimitiveTopology::ePointList)
                              .setPrimitiveRestartEnable(vk::False);

    // vk::Viewport viewport{0.0f,
    //                       0.0f,
    //                       static_cast<float>(swap_chain_extent.width),
    //                       static_cast<float>(swap_chain_extent.height),
    //                       0.0f,
    //                       1.0f};

    auto view_port_state = vk::PipelineViewportStateCreateInfo()
                               .setViewportCount(1)
                               .setScissorCount(1);

    auto rasterizer = vk::PipelineRasterizationStateCreateInfo()
                          .setDepthClampEnable(vk::False)
                          .setRasterizerDiscardEnable(vk::False)
                          .setPolygonMode(vk::PolygonMode::eFill)
                          .setCullMode(vk::CullModeFlagBits::eBack)
                          .setFrontFace(vk::FrontFace::eCounterClockwise)
                          .setDepthBiasEnable(vk::False)
                          .setLineWidth(1.0f);

    auto multisampling =
        vk::PipelineMultisampleStateCreateInfo()
            .setRasterizationSamples(vk::SampleCountFlagBits::e1)
            .setSampleShadingEnable(vk::False);

    auto color_blend_attachment =
        vk::PipelineColorBlendAttachmentState()
            .setBlendEnable(vk::True)
            .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
            .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
            .setColorBlendOp(vk::BlendOp::eAdd)
            .setSrcAlphaBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
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

    std::vector<vk::DynamicState> dynamic_states = {vk::DynamicState::eViewport,
                                                    vk::DynamicState::eScissor};

    auto dynamic_state =
        vk::PipelineDynamicStateCreateInfo()
            .setDynamicStateCount(static_cast<uint32_t>(dynamic_states.size()))
            .setPDynamicStates(dynamic_states.data());

    vk::PipelineLayoutCreateInfo pipeline_layout_info;

    pipeline_layout = vk::raii::PipelineLayout(device, pipeline_layout_info);

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

  void create_compute_pipeline() {
    vk::raii::ShaderModule shader_module =
        create_shader_module(read_file("shaders/slang.spv"));

    auto compute_shader_stage_info =
        vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eCompute)
            .setModule(shader_module)
            .setPName("compMain");

    auto pipeline_layout_info =
        vk::PipelineLayoutCreateInfo().setSetLayoutCount(1).setPSetLayouts(
            &*compute_descriptor_set_layout);

    compute_pipeline_layout =
        vk::raii::PipelineLayout(device, pipeline_layout_info);

    auto pipeline_info = vk::ComputePipelineCreateInfo()
                             .setStage(compute_shader_stage_info)
                             .setLayout(*compute_pipeline_layout);

    compute_pipeline = vk::raii::Pipeline(device, nullptr, pipeline_info);
  }

  void create_command_pool() {
    auto pool_info =
        vk::CommandPoolCreateInfo()
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
            .setQueueFamilyIndex(queue_index);
    command_pool = vk::raii::CommandPool(device, pool_info);
  }

  void create_shader_storage_buffers() {
    std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
    std::uniform_real_distribution rndDist(0.0f, 1.0f);

    std::vector<Particle> particles(PARTICLE_COUNT);
    for (auto &particle : particles) {
      float r = 0.25f * sqrtf(rndDist(rndEngine));
      float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
      float x = r * cosf(theta) * HEIGHT / WIDTH;
      float y = r * sinf(theta);
      particle.pos = glm::vec2(x, y);
      particle.vel = glm::normalize(glm::vec2(x, y)) * 0.00025f;
      particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine),
                                 rndDist(rndEngine), 1.0f);
    }

    vk::DeviceSize buffer_size = sizeof(Particle) * PARTICLE_COUNT;

    vk::raii::Buffer staging_buffer({});
    vk::raii::DeviceMemory stagin_buffer_memory({});

    create_buffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc,
                  vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent,
                  staging_buffer, stagin_buffer_memory);

    void *data_staging = stagin_buffer_memory.mapMemory(0, buffer_size);
    memcpy(data_staging, particles.data(), (size_t)buffer_size);
    stagin_buffer_memory.unmapMemory();

    shader_storage_buffers.clear();
    shared_storage_buffers_memory.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::raii::Buffer shader_storage_buffer_temp({});
      vk::raii::DeviceMemory shader_storage_buffer_temp_memory({});
      create_buffer(buffer_size,
                    vk::BufferUsageFlagBits::eStorageBuffer |
                        vk::BufferUsageFlagBits::eVertexBuffer |
                        vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal,
                    shader_storage_buffer_temp,
                    shader_storage_buffer_temp_memory);
      copy_buffer(staging_buffer, shader_storage_buffer_temp, buffer_size);
      shader_storage_buffers.emplace_back(
          std::move(shader_storage_buffer_temp));
      shared_storage_buffers_memory.emplace_back(
          std::move(shader_storage_buffer_temp_memory));
    }
  }

  void create_uniform_buffers() {
    uniform_buffers.clear();
    uniform_buffers_memory.clear();
    uniform_buffers_mapped.clear();

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DeviceSize buffer_size = sizeof(UniformBufferObject);
      vk::raii::Buffer buffer({});
      vk::raii::DeviceMemory buffer_memory({});
      create_buffer(buffer_size, vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent,
                    buffer, buffer_memory);

      uniform_buffers.emplace_back(std::move(buffer));
      uniform_buffers_memory.emplace_back(std::move(buffer_memory));
      uniform_buffers_mapped.emplace_back(
          uniform_buffers_memory[i].mapMemory(0, buffer_size));
    }
  }

  void create_descriptor_pool() {
    std::array<vk::DescriptorPoolSize, 2> pool_size{
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eUniformBuffer)
            .setDescriptorCount(MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(MAX_FRAMES_IN_FLIGHT)};

    auto pool_info =
        vk::DescriptorPoolCreateInfo()
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
            .setMaxSets(MAX_FRAMES_IN_FLIGHT)
            .setPoolSizeCount(pool_size.size())
            .setPPoolSizes(pool_size.data());

    descriptor_pool = vk::raii::DescriptorPool(device, pool_info);
  };

  void create_compute_descriptor_sets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                 compute_descriptor_set_layout);
    auto alloc_info = vk::DescriptorSetAllocateInfo()
                          .setDescriptorPool(*descriptor_pool)
                          .setDescriptorSetCount(MAX_FRAMES_IN_FLIGHT)
                          .setPSetLayouts(layouts.data());

    compute_descriptor_sets.clear();
    compute_descriptor_sets = device.allocateDescriptorSets(alloc_info);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DescriptorBufferInfo buffer_info(uniform_buffers[i], 0,
                                           sizeof(UniformBufferObject));

      vk::DescriptorBufferInfo storage_buffer_info_last_frame(
          shader_storage_buffers[(i - 1) % MAX_FRAMES_IN_FLIGHT], 0,
          sizeof(Particle) * PARTICLE_COUNT);

      vk::DescriptorBufferInfo storage_buffer_info_current_frame(
          shader_storage_buffers[i], 0, sizeof(Particle) * PARTICLE_COUNT);

      std::array<vk::WriteDescriptorSet, 3> descriptor_writes{
          vk::WriteDescriptorSet()
              .setDstSet(*compute_descriptor_sets[i])
              .setDstBinding(0)
              .setDstArrayElement(0)
              .setDescriptorCount(1)
              .setDescriptorType(vk::DescriptorType::eUniformBuffer)
              .setPImageInfo(nullptr)
              .setPBufferInfo(&buffer_info)
              .setPTexelBufferView(nullptr),
          vk::WriteDescriptorSet()
              .setDstSet(*compute_descriptor_sets[i])
              .setDstBinding(1)
              .setDstArrayElement(0)
              .setDescriptorCount(1)
              .setDescriptorType(vk::DescriptorType::eStorageBuffer)
              .setPImageInfo(nullptr)
              .setPBufferInfo(&storage_buffer_info_last_frame)
              .setPTexelBufferView(nullptr),
          vk::WriteDescriptorSet()
              .setDstSet(*compute_descriptor_sets[i])
              .setDstBinding(2)
              .setDstArrayElement(0)
              .setDescriptorCount(1)
              .setDescriptorType(vk::DescriptorType::eStorageBuffer)
              .setPImageInfo(nullptr)
              .setPBufferInfo(&storage_buffer_info_current_frame)
              .setPTexelBufferView(nullptr),
      };
      device.updateDescriptorSets(descriptor_writes, {});
    }
  }

  void create_buffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                     vk::MemoryPropertyFlags properties,
                     vk::raii::Buffer &buffer,
                     vk::raii::DeviceMemory &buffer_memory) const {

    auto buffer_info =
        vk::BufferCreateInfo().setSize(size).setUsage(usage).setSharingMode(
            vk::SharingMode::eExclusive);

    buffer = vk::raii::Buffer(device, buffer_info);

    auto mem_requirements = buffer.getMemoryRequirements();
    auto alloc_info = vk::MemoryAllocateInfo()
                          .setAllocationSize(mem_requirements.size)
                          .setMemoryTypeIndex(find_memory_type(
                              mem_requirements.memoryTypeBits, properties));

    buffer_memory = vk::raii::DeviceMemory(device, alloc_info);
    buffer.bindMemory(buffer_memory, 0);
  }

  [[nodiscard]] vk::raii::CommandBuffer begin_single_time_commands() const {
    auto alloc_info = vk::CommandBufferAllocateInfo()
                          .setCommandPool(*command_pool)
                          .setLevel(vk::CommandBufferLevel::ePrimary)
                          .setCommandBufferCount(1);

    auto command_buffer =
        std::move(vk::raii::CommandBuffers(device, alloc_info).front());
    auto begin_info = vk::CommandBufferBeginInfo().setFlags(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    command_buffer.begin(begin_info);

    return command_buffer;
  }

  void end_single_time_commands(
      const vk::raii::CommandBuffer &command_buffer) const {
    command_buffer.end();

    auto submit_info =
        vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(
            &*command_buffer);
    queue.submit(submit_info, nullptr);
    queue.waitIdle();
  }

  void copy_buffer(const vk::raii::Buffer &src_buffer,
                   const vk::raii::Buffer &dst_buffer,
                   vk::DeviceSize size) const {
    auto command_buffer = begin_single_time_commands();
    command_buffer.copyBuffer(
        src_buffer, dst_buffer,
        vk::BufferCopy().setSrcOffset(0).setDstOffset(0).setSize(size));
    end_single_time_commands(command_buffer);
  }

  [[nodiscard]] uint32_t
  find_memory_type(uint32_t type_filter,
                   vk::MemoryPropertyFlags properties) const {
    auto mem_properties = physical_device.getMemoryProperties();

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
      if ((type_filter & (1 << i)) &&
          (mem_properties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }
    throw std::runtime_error("failed to find suitable memory type!");
  }

  void create_command_buffers() {
    command_buffers.clear();
    auto alloc_info = vk::CommandBufferAllocateInfo()
                          .setCommandPool(*command_pool)
                          .setLevel(vk::CommandBufferLevel::ePrimary)
                          .setCommandBufferCount(MAX_FRAMES_IN_FLIGHT);
    command_buffers = vk::raii::CommandBuffers(device, alloc_info);
  }

  void create_compute_command_buffers() {
    compute_command_buffers.clear();

    auto alloc_info = vk::CommandBufferAllocateInfo()
                          .setCommandPool(*command_pool)
                          .setLevel(vk::CommandBufferLevel::ePrimary)
                          .setCommandBufferCount(MAX_FRAMES_IN_FLIGHT);

    compute_command_buffers = vk::raii::CommandBuffers(device, alloc_info);
  }

  void record_command_buffer(uint32_t image_index) {
    auto &command_buffer = command_buffers[frame_index];
    command_buffer.reset();
    command_buffer.begin({});

    transition_image_layour(image_index, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {},
                            vk::AccessFlagBits2::eColorAttachmentWrite,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput);

    vk::ClearValue clear_color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    auto attachment_info =
        vk::RenderingAttachmentInfo()
            .setImageView(swap_chain_image_views[image_index])
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(clear_color);

    auto rendering_info =
        vk::RenderingInfo()
            .setRenderArea(
                vk::Rect2D().setOffset({0, 0}).setExtent(swap_chain_extent))
            .setLayerCount(1)
            .setColorAttachmentCount(1)
            .setPColorAttachments(&attachment_info);

    command_buffer.beginRendering(rendering_info);
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                *graphics_pipeline);
    command_buffer.setViewport(
        0,
        vk::Viewport(0.0f, 0.0f, static_cast<float>(swap_chain_extent.width),
                     static_cast<float>(swap_chain_extent.height), 0.0f, 1.0f));

    command_buffer.setScissor(
        0, vk::Rect2D(vk::Offset2D(0, 0), swap_chain_extent));

    command_buffer.bindVertexBuffers(0, {shader_storage_buffers[frame_index]},
                                     {0});

    command_buffer.draw(PARTICLE_COUNT, 1, 0, 0);
    command_buffer.endRendering();

    transition_image_layour(image_index,
                            vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::ePresentSrcKHR,
                            vk::AccessFlagBits2::eColorAttachmentWrite, {},
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits2::eBottomOfPipe);

    command_buffer.end();
  }

  void transition_image_layour(uint32_t image_index, vk::ImageLayout old_layout,
                               vk::ImageLayout new_layout,
                               vk::AccessFlags2 src_access_mask,
                               vk::AccessFlags2 dst_access_mask,
                               vk::PipelineStageFlags2 src_stage_mask,
                               vk::PipelineStageFlags2 dst_stage_mask) {
    auto barrier = vk::ImageMemoryBarrier2()
                       .setSrcStageMask(src_stage_mask)
                       .setSrcAccessMask(src_access_mask)
                       .setDstStageMask(dst_stage_mask)
                       .setDstAccessMask(dst_access_mask)
                       .setOldLayout(old_layout)
                       .setNewLayout(new_layout)
                       .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                       .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                       .setImage(swap_chain_images[image_index])
                       .setSubresourceRange(
                           vk::ImageSubresourceRange()
                               .setAspectMask(vk::ImageAspectFlagBits::eColor)
                               .setBaseMipLevel(0)
                               .setLevelCount(1)
                               .setBaseArrayLayer(0)
                               .setLayerCount(1));

    auto dependency_info = vk::DependencyInfo()
                               .setDependencyFlags({})
                               .setImageMemoryBarrierCount(1)
                               .setPImageMemoryBarriers(&barrier);

    command_buffers[frame_index].pipelineBarrier2(dependency_info);
  }

  void record_compute_command_buffer() {
    auto &command_buffer = compute_command_buffers[frame_index];
    command_buffer.reset();
    command_buffer.begin({});
    command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                compute_pipeline);
    command_buffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, compute_pipeline_layout, 0,
        {compute_descriptor_sets[frame_index]}, {});

    command_buffer.dispatch(PARTICLE_COUNT / 256, 1, 1);
    command_buffer.end();
  }

  void create_sync_objects() {
    in_flight_fences.clear();

    auto semaphore_type = vk::SemaphoreTypeCreateInfo()
                              .setSemaphoreType(vk::SemaphoreType::eTimeline)
                              .setInitialValue(0);

    semaphore = vk::raii::Semaphore(
        device, vk::SemaphoreCreateInfo().setPNext(&semaphore_type));
    time_line_value = 0;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::FenceCreateInfo fence_info{};
      in_flight_fences.emplace_back(device, fence_info);
    }
  }

  void update_uniform_buffer(uint32_t current_image) {
    UniformBufferObject ubo{};
    ubo.delta_time = static_cast<float>(last_frame_time) * 2.0f;

    memcpy(uniform_buffers_mapped[current_image], &ubo, sizeof(ubo));
  }

  void draw_frame() {
    auto [result, image_index] = swap_chain.acquireNextImage(
        UINT64_MAX, nullptr, *in_flight_fences[frame_index]);

    auto fence_result = device.waitForFences(*in_flight_fences[frame_index],
                                             vk::True, UINT64_MAX);

    if (fence_result != vk::Result::eSuccess)
      throw std::runtime_error("failed to wait for fence!");
    device.resetFences(*in_flight_fences[frame_index]);

    uint64_t compute_wait_value = time_line_value;
    uint64_t compute_signal_value = ++time_line_value;
    uint64_t graphics_wait_value = compute_signal_value;
    uint64_t graphics_signal_value = ++time_line_value;

    update_uniform_buffer(frame_index);

    {
      record_compute_command_buffer();
      auto compute_timeline_info =
          vk::TimelineSemaphoreSubmitInfo()
              .setWaitSemaphoreValueCount(1)
              .setPWaitSemaphoreValues(&compute_wait_value)
              .setSignalSemaphoreValueCount(1)
              .setPSignalSemaphoreValues(&compute_signal_value);

      vk::PipelineStageFlags wait_stages[] = {
          vk::PipelineStageFlagBits::eComputeShader};

      auto submit_info =
          vk::SubmitInfo()
              .setPNext(&compute_timeline_info)
              .setWaitSemaphoreCount(1)
              .setPWaitSemaphores(&*semaphore)
              .setPWaitDstStageMask(wait_stages)
              .setCommandBufferCount(1)
              .setPCommandBuffers(&*compute_command_buffers[frame_index])
              .setSignalSemaphoreCount(1)
              .setPSignalSemaphores(&*semaphore);

      queue.submit(submit_info, nullptr);
    }
    {
      record_command_buffer(image_index);

      vk::PipelineStageFlags wait_stages =
          vk::PipelineStageFlagBits::eVertexInput;

      auto graphics_time_line_info =
          vk::TimelineSemaphoreSubmitInfo()
              .setWaitSemaphoreValueCount(1)
              .setPWaitSemaphoreValues(&graphics_wait_value)
              .setSignalSemaphoreValueCount(1)
              .setPSignalSemaphoreValues(&graphics_signal_value);

      auto graphics_submit_info =
          vk::SubmitInfo()
              .setPNext(&graphics_time_line_info)
              .setWaitSemaphoreCount(1)
              .setPWaitSemaphores(&*semaphore)
              .setPWaitDstStageMask(&wait_stages)
              .setCommandBufferCount(1)
              .setPCommandBuffers(&*command_buffers[frame_index])
              .setSignalSemaphoreCount(1)
              .setPSignalSemaphores(&*semaphore);

      queue.submit(graphics_submit_info, nullptr);

      auto wait_info = vk::SemaphoreWaitInfo()
                           .setSemaphoreCount(1)
                           .setPSemaphores(&*semaphore)
                           .setPValues(&graphics_signal_value);

      auto result = device.waitSemaphores(wait_info, UINT64_MAX);
      if (result != vk::Result::eSuccess)
        throw std::runtime_error("failed to wait for semaphore!");

      const auto present_info_khr = vk::PresentInfoKHR()
                                        .setWaitSemaphoreCount(0)
                                        .setPWaitSemaphores(nullptr)
                                        .setSwapchainCount(1)
                                        .setPSwapchains(&*swap_chain)
                                        .setPImageIndices(&image_index);

      result = queue.presentKHR(present_info_khr);

      if ((result == vk::Result::eSuboptimalKHR) ||
          (result == vk::Result::eErrorOutOfDateKHR) || framebuffer_resized) {
        framebuffer_resized = false;
        recreate_swap_chain();
      } else {
        assert(result == vk::Result::eSuccess);
      }
    }
    frame_index = (frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  [[nodiscard]] vk::raii::ShaderModule
  create_shader_module(const std::vector<char> &code) const {
    auto create_info =
        vk::ShaderModuleCreateInfo()
            .setCodeSize(code.size())
            .setPCode(reinterpret_cast<const uint32_t *>(code.data()));

    auto shader_module = vk::raii::ShaderModule{device, create_info};

    return shader_module;
  }

  static uint32_t choose_swap_min_image_count(
      vk::SurfaceCapabilitiesKHR const &surface_capabilities) {
    auto min_image_count = std::max(3u, surface_capabilities.minImageCount);
    if ((0 < surface_capabilities.maxImageCount) &&
        (surface_capabilities.maxImageCount < min_image_count))
      min_image_count = surface_capabilities.maxImageCount;
    return min_image_count;
  }

  static vk::SurfaceFormatKHR choose_swap_surface_format(
      const std::vector<vk::SurfaceFormatKHR> &available_formats) {
    assert(!available_formats.empty());
    const auto format_it = std::find_if(
        available_formats.begin(), available_formats.end(),
        [](const vk::SurfaceFormatKHR &format) {
          return format.format == vk::Format::eR8G8B8A8Srgb &&
                 format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });
    return format_it != available_formats.end() ? *format_it
                                                : available_formats[0];
  }

  static vk::PresentModeKHR choose_swap_present_mode(
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
        std::clamp<uint32_t>(height, capabilities.minImageExtent.height,
                             capabilities.maxImageExtent.height),
    };
  }

  [[nodiscard]] std::vector<const char *> get_required_instance_extensions() {
    uint32_t glfw_ext_count = 0;
    auto glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    std::vector extensions(glfw_exts, glfw_exts + glfw_ext_count);
    if (enable_validation_layers) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    return extensions;
  }

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL
  debug_callback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                 vk::DebugUtilsMessageTypeFlagsEXT type,
                 const vk::DebugUtilsMessengerCallbackDataEXT *pCall_back_data,
                 void *pUser_data) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
        severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
      std::cerr << "validation layer: type " << to_string(type)
                << " msg: " << pCall_back_data->pMessage << std::endl;
    }

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
    ComputeShaderApplication app;
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
