#include "vulkan/vulkan.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/glm.hpp>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
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
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  static vk::VertexInputBindingDescription get_binding_description() {
    return vk::VertexInputBindingDescription()
        .setBinding(0)
        .setStride(sizeof(Vertex))
        .setInputRate(vk::VertexInputRate::eVertex);
  }

  static std::array<vk::VertexInputAttributeDescription, 2>
  get_attribute_descriptio() {
    return {
        vk::VertexInputAttributeDescription()
            .setLocation(0)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32Sfloat)
            .setOffset(offsetof(Vertex, pos)),
        vk::VertexInputAttributeDescription()
            .setLocation(1)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32B32Sfloat)
            .setOffset(offsetof(Vertex, color)),
    };
  }
};

const std::vector<Vertex> vertices = {{{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
                                      {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
                                      {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};

class HelloTriangleApplication {
public:
  void run() {
    printf("Ho completato la parte sulla graphics pipeline basics!\n");
    printf("Ref: "
           "https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/"
           "02_Graphics_pipeline_basics/00_Introduction.html\n");
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
  uint32_t queue_index = ~0;
  vk::raii::Queue queue = nullptr;
  vk::raii::SwapchainKHR swap_chain = nullptr;
  std::vector<vk::Image> swap_chain_images;
  vk::SurfaceFormatKHR swap_chain_surface_format;
  vk::Extent2D swap_chain_extent;
  std::vector<vk::raii::ImageView> swap_chain_image_views;
  vk::raii::PipelineLayout pipeline_layout = nullptr;
  vk::raii::Pipeline graphics_pipeline = nullptr;
  vk::raii::CommandPool command_pool = nullptr;
  vk::raii::Buffer vertex_buffer = nullptr;
  vk::raii::DeviceMemory vertex_buffer_memory = nullptr;
  std::vector<vk::raii::CommandBuffer> command_buffers;
  std::vector<vk::raii::Semaphore> present_complete_semaphores;
  std::vector<vk::raii::Semaphore> render_finished_semaphores;
  std::vector<vk::raii::Fence> in_flight_fences;
  uint32_t frame_index = 0;

  bool framebuffer_resized = false;
  std::vector<const char *> required_device_extension = {
      vk::KHRSwapchainExtensionName};

  void init_window() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vukan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
  }

  static void framebuffer_resize_callback(GLFWwindow *window, int width,
                                          int height) {
    auto app = reinterpret_cast<HelloTriangleApplication *>(
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
    create_graphics_pipeline();
    create_command_pool();
    create_vertex_buffer();
    create_command_buffers();
    create_sync_objects();
  }

  void main_loop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      draw_frame();
    }

    device.waitIdle();
  }

  void draw_frame() {
    auto fence_result = device.waitForFences(*in_flight_fences[frame_index],
                                             vk::True, UINT64_MAX);
    if (fence_result != vk::Result::eSuccess)
      throw std::runtime_error("failed to wait for fence!");

    auto [result, image_index] = swap_chain.acquireNextImage(
        UINT64_MAX, *present_complete_semaphores[frame_index], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreate_swap_chain();
      return;
    }
    if (result != vk::Result::eSuccess &&
        result != vk::Result::eSuboptimalKHR) {
      assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
      throw std::runtime_error("failed to acquire swap chain image");
    }
    device.resetFences(*in_flight_fences[frame_index]);

    command_buffers[frame_index].reset();
    record_command_buffer(image_index);

    vk::PipelineStageFlags wait_destination_stage_mask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);

    const auto submit_info =
        vk::SubmitInfo()
            .setWaitSemaphoreCount(1)
            .setPWaitSemaphores(&*present_complete_semaphores[frame_index])
            .setPWaitDstStageMask(&wait_destination_stage_mask)
            .setCommandBufferCount(1)
            .setPCommandBuffers(&*command_buffers[frame_index])
            .setSignalSemaphoreCount(1)
            .setPSignalSemaphores(&*render_finished_semaphores[frame_index]);

    queue.submit(submit_info, *in_flight_fences[frame_index]);

    /* This is a Subpass Dependency. It is optional and not reproduces in the
  demo code auto dependency = vk::SubpassDependency()
            .setSrcSubpass(vk::SubpassExternal)
            .setDstSubpass(0)
            .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setSrcAccessMask(vk::AccessFlagBits::eNone)
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

  vk::RenderPassInfo render_pass_info; //Add in the members of class
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;
  */

    const auto present_info_khr =
        vk::PresentInfoKHR()
            .setWaitSemaphoreCount(1)
            .setPWaitSemaphores(&*render_finished_semaphores[frame_index])
            .setSwapchainCount(1)
            .setPSwapchains(&*swap_chain)
            .setPImageIndices(&image_index)
            .setPResults(nullptr);

    result = queue.presentKHR(present_info_khr);

    if ((result == vk::Result::eSuboptimalKHR) ||
        (result == vk::Result::eErrorOutOfDateKHR)) {
      framebuffer_resized = false;
      recreate_swap_chain();
    } else {
      assert(result == vk::Result::eSuccess);
    }
    frame_index = (frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
  }
  void create_sync_objects() {
    assert(present_complete_semaphores.empty() &&
           render_finished_semaphores.empty() && in_flight_fences.empty());
    for (int i = 0; i < swap_chain_images.size(); i++) {
      render_finished_semaphores.emplace_back(device,
                                              vk::SemaphoreCreateInfo());
    }
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      present_complete_semaphores.emplace_back(device,
                                               vk::SemaphoreCreateInfo());
      in_flight_fences.emplace_back(
          device,
          vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled));
    }
  }

  void cleanup() {
    cleanup_swapchain();
    glfwDestroyWindow(window);

    glfwTerminate();
  }

  void cleanup_swapchain() {
    swap_chain.clear();
    swap_chain = nullptr;
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
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

    bool supports_required_features =
        features.get<vk::PhysicalDeviceVulkan11Features>()
            .shaderDrawParameters &&
        features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
        features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
        features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState;

    return supports_required_features && supported_all_required_extensions &&
           supported_vulkan1_3 && supports_graphics;
  }

  void create_surface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
      throw std::runtime_error("failed to create window surface!");
    }
    std::cout << "Surfece creata" << std::endl;
    surface = vk::raii::SurfaceKHR(instance, _surface);
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

    auto feature_chain =
        vk::StructureChain<vk::PhysicalDeviceFeatures2,
                           vk::PhysicalDeviceVulkan11Features,
                           vk::PhysicalDeviceVulkan13Features,
                           vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>{
            vk::PhysicalDeviceFeatures2{},
            vk::PhysicalDeviceVulkan11Features().setShaderDrawParameters(true),
            vk::PhysicalDeviceVulkan13Features()
                .setSynchronization2(true)
                .setDynamicRendering(true),
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
    queue = vk::raii::Queue(device, queue_index, 0);
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
            .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    for (auto &image : swap_chain_images) {
      image_view_create_info.image = image;
      swap_chain_image_views.emplace_back(device, image_view_create_info);
    }
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

    auto binding_description = Vertex::get_binding_description();
    auto attribute_description = Vertex::get_attribute_descriptio();
    auto vertex_input_info =
        vk::PipelineVertexInputStateCreateInfo()
            .setVertexBindingDescriptionCount(1)
            .setPVertexBindingDescriptions(&binding_description)
            .setVertexAttributeDescriptionCount(
                static_cast<uint32_t>(attribute_description.size()))
            .setPVertexAttributeDescriptions(attribute_description.data());
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

  void create_command_pool() {
    auto pool_info =
        vk::CommandPoolCreateInfo()
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
            .setQueueFamilyIndex(queue_index);
    command_pool = vk::raii::CommandPool(device, pool_info);
  }

  void create_vertex_buffer() {
    auto buffer_info = vk::BufferCreateInfo()
                           .setSize(sizeof(vertices[0]) * vertices.size())
                           .setUsage(vk::BufferUsageFlagBits::eVertexBuffer)
                           .setSharingMode(vk::SharingMode::eExclusive);

    vertex_buffer = vk::raii::Buffer(device, buffer_info);

    auto mem_requirements = vertex_buffer.getMemoryRequirements();
    auto mem_allocate_info =
        vk::MemoryAllocateInfo()
            .setAllocationSize(mem_requirements.size)
            .setMemoryTypeIndex(find_memory_type(
                mem_requirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent));

    vertex_buffer_memory = vk::raii::DeviceMemory(device,mem_allocate_info);
    vertex_buffer.bindMemory(*vertex_buffer_memory, 0);

    void *data = vertex_buffer_memory.mapMemory(0, buffer_info.size);
    memcpy(data, vertices.data(), buffer_info.size);
    vertex_buffer_memory.unmapMemory();
  }

  uint32_t find_memory_type(uint32_t type_filter,
                            vk::MemoryPropertyFlags properties) {
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
    auto alloc_info = vk::CommandBufferAllocateInfo()
                          .setCommandPool(command_pool)
                          .setLevel(vk::CommandBufferLevel::ePrimary)
                          .setCommandBufferCount(MAX_FRAMES_IN_FLIGHT);
    command_buffers = vk::raii::CommandBuffers(device, alloc_info);
  }

  void record_command_buffer(uint32_t image_index) {
    command_buffers[frame_index].begin({});

    transition_image_layour(image_index, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {},
                            vk::AccessFlagBits2::eColorAttachmentWrite,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput);

    auto clear_color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);

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

    command_buffers[frame_index].beginRendering(rendering_info);
    command_buffers[frame_index].bindPipeline(vk::PipelineBindPoint::eGraphics,
                                              *graphics_pipeline);
    command_buffers[frame_index].bindVertexBuffers(0, *vertex_buffer, {0});
    command_buffers[frame_index].setViewport(
        0,
        vk::Viewport(0.0f, 0.0f, static_cast<float>(swap_chain_extent.width),
                     static_cast<float>(swap_chain_extent.height), 0.0f, 1.0f));
    command_buffers[frame_index].setScissor(
        0, vk::Rect2D(vk::Offset2D(0, 0), swap_chain_extent));
    command_buffers[frame_index].draw(static_cast<uint32_t>(vertices.size()), 1, 0, 0);
    command_buffers[frame_index].endRendering();

    transition_image_layour(image_index,
                            vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::ePresentSrcKHR,
                            vk::AccessFlagBits2::eColorAttachmentWrite, {},
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits2::eBottomOfPipe);

    command_buffers[frame_index].end();
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
        std::clamp<uint32_t>(height, capabilities.minImageExtent.height,
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
