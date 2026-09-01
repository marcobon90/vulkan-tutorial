#include "vulkan/vulkan.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/fwd.hpp>
#include <memory>
#include <ostream>
#include <tuple>
#include <utility>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
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

#define GLFW_EXPOSE_NATIVE_WIN32

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validation_layers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enable_validation_layers = false;
#else
constexpr bool enable_validation_layers = true;
#endif

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

struct AppInfo {
  bool dynamic_rendering_supported = false;
  bool timeline_sempaphore_supported = false;
  bool synchronization2_supported = false;
};

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 text_coord;

  static vk::VertexInputBindingDescription get_binding_description() {
    return vk::VertexInputBindingDescription()
        .setBinding(0)
        .setStride(sizeof(Vertex))
        .setInputRate(vk::VertexInputRate::eVertex);
  }

  static std::array<vk::VertexInputAttributeDescription, 3>
  get_attribute_descriptio() {
    return {
        vk::VertexInputAttributeDescription()
            .setLocation(0)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32B32Sfloat)
            .setOffset(offsetof(Vertex, pos)),
        vk::VertexInputAttributeDescription()
            .setLocation(1)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32B32Sfloat)
            .setOffset(offsetof(Vertex, color)),
        vk::VertexInputAttributeDescription()
            .setLocation(2)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32Sfloat)
            .setOffset(offsetof(Vertex, text_coord)),
    };
  }

  bool operator==(const Vertex &other) const {
    return pos == other.pos && color == other.color &&
           text_coord == other.text_coord;
  }
};

namespace std {
template <> struct hash<Vertex> {
  size_t operator()(Vertex const &vertex) const {
    return ((hash<glm::vec3>()(vertex.pos) ^
             (hash<glm::vec3>()(vertex.color) << 1)) >>
            1) ^
           (hash<glm::vec2>()(vertex.text_coord) << 1);
  }
};
} // namespace std

/*
 * Use alignas(16) because std140 memory in Vulkan shader.
 */
struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

class HelloTriangleApplication {
public:
  void run() {
    init_window();
    init_vulkan();
    main_loop();
    cleanup();
  }

private:
  AppInfo app_info;

  GLFWwindow *window = nullptr;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debug_messenger = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physical_device = nullptr;
  vk::SampleCountFlagBits msaa_samples = vk::SampleCountFlagBits::e1;
  vk::raii::Device device = nullptr;
  uint32_t queue_index = ~0;
  vk::raii::Queue queue = nullptr;
  vk::raii::SwapchainKHR swap_chain = nullptr;
  std::vector<vk::Image> swap_chain_images;
  vk::SurfaceFormatKHR swap_chain_surface_format;
  vk::Extent2D swap_chain_extent;
  std::vector<vk::raii::ImageView> swap_chain_image_views;

  // Traditional render pass (fallback for non-dynamic rendering)
  vk::raii::RenderPass render_pass = nullptr;
  std::vector<vk::raii::Framebuffer> swap_chain_frame_buffers;

  vk::raii::DescriptorSetLayout descriptor_set_layout = nullptr;
  vk::raii::PipelineLayout pipeline_layout = nullptr;
  vk::raii::Pipeline graphics_pipeline = nullptr;

  vk::raii::Image color_image = nullptr;
  vk::raii::DeviceMemory color_image_memory = nullptr;
  vk::raii::ImageView color_image_view = nullptr;

  vk::raii::Image depth_image = nullptr;
  vk::raii::DeviceMemory depth_image_memory = nullptr;
  vk::raii::ImageView depth_image_view = nullptr;

  uint32_t mip_levels = 0;
  vk::raii::Image texture_image = nullptr;
  vk::raii::DeviceMemory texture_image_memory = nullptr;
  vk::raii::ImageView texture_image_view = nullptr;
  vk::raii::Sampler texture_sampler = nullptr;

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  vk::raii::Buffer vertex_buffer = nullptr;
  vk::raii::DeviceMemory vertex_buffer_memory = nullptr;
  vk::raii::Buffer index_buffer = nullptr;
  vk::raii::DeviceMemory index_buffer_memory = nullptr;

  std::vector<vk::raii::Buffer> uniform_buffers;
  std::vector<vk::raii::DeviceMemory> uniform_buffers_memory;
  std::vector<void *> uniform_buffers_mapped;

  vk::raii::DescriptorPool descriptor_pool = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptor_sets;

  vk::raii::CommandPool command_pool = nullptr;
  std::vector<vk::raii::CommandBuffer> command_buffers;

  std::vector<vk::raii::Semaphore> present_complete_semaphores;
  std::vector<vk::raii::Semaphore> render_finished_semaphores;
  std::vector<vk::raii::Fence> in_flight_fences;
  uint32_t frame_index = 0;
  vk::raii::Semaphore time_line_semaphore = nullptr;
  uint64_t time_line_value = 0;

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
    detect_feature_support();
    create_logical_device();
    create_swap_chain();
    create_image_views();

    if (!app_info.dynamic_rendering_supported) {
      create_render_pass();
      create_frame_buffers();
    }

    create_descriptor_set_layout();
    create_graphics_pipeline();
    create_command_pool();
    create_color_resources();
    create_depth_resources();
    create_texture_image();
    create_texture_image_view();
    create_texture_sampler();
    load_model();
    create_vertex_buffer();
    create_index_buffer();
    create_uniform_buffers();
    create_descriptor_pool();
    create_descriptor_sets();
    create_command_buffers();
    create_sync_objects();

    // Print feature support summary
    std::cout << "\nFeature support summary:\n";
    std::cout << "- Dynamic rendering: "
              << (app_info.dynamic_rendering_supported ? "Yes" : "No") << "\n";
    std::cout << "- Timeline Semaphore: "
              << (app_info.timeline_sempaphore_supported ? "Yes" : "No")
              << "\n";
    std::cout << "- Synchronization2: "
              << (app_info.synchronization2_supported ? "Yes" : "No") << "\n";
  }

  void main_loop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      draw_frame();
    }

    device.waitIdle();
  }

  void cleanup_swapchain() {
    swap_chain_frame_buffers.clear();
    swap_chain_image_views.clear();
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

    if (!app_info.dynamic_rendering_supported) {
      create_render_pass();
      create_frame_buffers();
    }

    create_color_resources();
    create_depth_resources();
  }

  void create_instance() {
    constexpr vk::ApplicationInfo appInfo =
        vk::ApplicationInfo()
            .setPApplicationName("Hello Triangle")
            .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
            .setPEngineName("No Engine")
            .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
            .setApiVersion(vk::ApiVersion14);

    // Get  the required extensions
    auto required_exts = get_required_instance_extensions();

    /*
     * I deleted enable validation layers and properties instructions.
     * I enabled VK_LAYER_KHRONOS_validation layer using
     * - vkconfig.exe on Windows
     * - vkconfig gui on Linux
     * Ref:
     * https://docs.vulkan.org/tutorial/latest/12_Ecosystem_Utilities_and_Compatibility.html
     */
    vk::InstanceCreateInfo const createInfo =
        vk::InstanceCreateInfo()
            .setPApplicationInfo(&appInfo)
            .setEnabledExtensionCount(
                static_cast<uint32_t>(required_exts.size()))
            .setPpEnabledExtensionNames(required_exts.data());

    instance = vk::raii::Instance(context, createInfo);
  }

  void setup_debug_messenger() {
    // Always set up the debug messenger
    // It will only be used if validation layers are enabled via vulkanconfig

    vk::DebugUtilsMessageSeverityFlagsEXT severity_flags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

    vk::DebugUtilsMessageTypeFlagsEXT message_type_flags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

    vk::DebugUtilsMessengerCreateInfoEXT debug_utils_create_info_ext =
        vk::DebugUtilsMessengerCreateInfoEXT()
            .setMessageSeverity(severity_flags)
            .setMessageType(message_type_flags)
            .setPfnUserCallback(&debug_callback);

    try {
      debug_messenger =
          instance.createDebugUtilsMessengerEXT(debug_utils_create_info_ext);
    } catch (vk::SystemError &err) {
      // If the debug utils extension is not available, this will fail
      // That's okay; it just means validation layers aren't enabled
      std::cout << "Debug messenger not available. Validation layers may not "
                   "be enabled."
                << std::endl;
    }
  }

  void create_surface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
      throw std::runtime_error("failed to create window surface!");
    }
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

    return supported_all_required_extensions && supported_vulkan1_3 &&
           supports_graphics;
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

    msaa_samples = get_max_usable_sample_count();
  }

  void detect_feature_support() {
    auto device_properties = physical_device.getProperties();

    auto available_extensions =
        physical_device.enumerateDeviceExtensionProperties();

    // Check for dynamic rendering support
    if (device_properties.apiVersion >= VK_VERSION_1_3) {
      app_info.dynamic_rendering_supported = true;
      std::cout << "Dynamic rendering supported via Vulkan 1.3\n";
    } else {
      // Check for the extension on older Vulkan versions
      for (const auto &extension : available_extensions) {
        if (strcmp(extension.extensionName,
                   VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) == 0) {
          app_info.dynamic_rendering_supported = true;
          std::cout << "Dynamic rendering supported via extensions\n";
          break;
        }
      }
    }

    // Check for timeline semaphores support
    if (device_properties.apiVersion >= VK_VERSION_1_2) {
      app_info.timeline_sempaphore_supported = true;
      std::cout << "Timeline semaphores supported via Vulkan 1.2\n";
    } else {
      // Check for the extension on older Vulkan versions
      for (const auto &extension : available_extensions) {
        if (strcmp(extension.extensionName,
                   VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0) {
          app_info.timeline_sempaphore_supported = true;
          std::cout << "Timeline semaphores supported via extensions\n";
          break;
        }
      }
    }

    // Check for dynamic rendering support
    if (device_properties.apiVersion >= VK_VERSION_1_3) {
      app_info.synchronization2_supported = true;
      std::cout << "Synchronization2 supported via Vulkan 1.3\n";
    } else {
      // Check for the extension on older Vulkan versions
      for (const auto &extension : available_extensions) {
        if (strcmp(extension.extensionName,
                   VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0) {
          app_info.synchronization2_supported = true;
          std::cout << "Synchronization2 supported via extensions\n";
          break;
        }
      }
    }
    // Add required extensions based on feature support
    if (app_info.dynamic_rendering_supported &&
        device_properties.apiVersion < VK_VERSION_1_3) {
      required_device_extension.push_back(
          VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    }

    if (app_info.timeline_sempaphore_supported &&
        device_properties.apiVersion < VK_VERSION_1_2) {
      required_device_extension.push_back(
          VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }

    if (app_info.synchronization2_supported &&
        device_properties.apiVersion < VK_VERSION_1_3) {
      required_device_extension.push_back(
          VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }
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

    // Create device with appropriate features
    auto features = physical_device.getFeatures2();

    // Setup feature chain based on detected support
    void *p_next = nullptr;

    // Add dynamic rendering if supported
    vk::PhysicalDeviceVulkan13Features vulkan13_features;
    vk::PhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features;

    if (app_info.dynamic_rendering_supported) {
      if (app_info.synchronization2_supported) {
        vulkan13_features.setDynamicRendering(vk::True)
            .setSynchronization2(vk::True)
            .setPNext(p_next);

        p_next = &vulkan13_features;
      } else {
        dynamic_rendering_features.setDynamicRendering(vk::True).setPNext(
            p_next);
        p_next = &dynamic_rendering_features;
      }
    }

    // Add timeline semaphores if supported
    vk::PhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features;
    if (app_info.timeline_sempaphore_supported) {
      timeline_semaphore_features.setTimelineSemaphore(vk::True).setPNext(
          p_next);
      p_next = &timeline_semaphore_features;
    }

    features.setPNext(p_next);

    // Create a device
    float queue_priority = 0.5f;
    auto device_queue_create_info = vk::DeviceQueueCreateInfo()
                                        .setQueueFamilyIndex(queue_index)
                                        .setQueueCount(1)
                                        .setPQueuePriorities(&queue_priority);

    auto device_create_info =
        vk::DeviceCreateInfo()
            .setPNext(&features)
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

    swap_chain_image_views.reserve(swap_chain_images.size());
    for (auto &image : swap_chain_images) {
      swap_chain_image_views.emplace_back(
          create_image_view(image, swap_chain_surface_format.format,
                            vk::ImageAspectFlagBits::eColor, 1));
    }
  }

  void create_render_pass() {
    if (app_info.dynamic_rendering_supported) {
      // No render pass needed with dynamic rendering
      std::cout << "Using dynamic rendering, skipping render pass creation\n";
      return;
    }

    std::cout << "Creating traditional render pass\n";

    auto color_attachment =
        vk::AttachmentDescription()
            .setFormat(swap_chain_surface_format.format)
            .setSamples(msaa_samples)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

    auto depth_attachment =
        vk::AttachmentDescription()
            .setFormat(find_depth_format())
            .setSamples(msaa_samples)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    auto color_attachment_resolve =
        vk::AttachmentDescription()
            .setFormat(swap_chain_surface_format.format)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setFinalLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    auto color_attachment_ref =
        vk::AttachmentReference().setAttachment(0).setLayout(
            vk::ImageLayout::eColorAttachmentOptimal);

    auto depth_attachment_ref =
        vk::AttachmentReference().setAttachment(1).setLayout(
            vk::ImageLayout::eDepthStencilAttachmentOptimal);

    auto color_attachment_resolve_ref =
        vk::AttachmentReference().setAttachment(2).setLayout(
            vk::ImageLayout::eColorAttachmentOptimal);

    auto subpass = vk::SubpassDescription()
                       .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                       .setColorAttachmentCount(1)
                       .setPColorAttachments(&color_attachment_ref)
                       .setPResolveAttachments(&color_attachment_resolve_ref)
                       .setPDepthStencilAttachment(&depth_attachment_ref);

    auto dependency =
        vk::SubpassDependency()
            .setSrcSubpass(VK_SUBPASS_EXTERNAL)
            .setDstSubpass(0)
            .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput |
                             vk::PipelineStageFlagBits::eEarlyFragmentTests)
            .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput |
                             vk::PipelineStageFlagBits::eEarlyFragmentTests)
            .setSrcAccessMask(vk::AccessFlagBits::eNone)
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite |
                              vk::AccessFlagBits::eDepthStencilAttachmentWrite);

    // Create the render pass
    std::array<vk::AttachmentDescription, 3> attachments = {
        color_attachment, depth_attachment, color_attachment_resolve};
    auto render_pass_info =
        vk::RenderPassCreateInfo()
            .setAttachmentCount(static_cast<uint32_t>(attachments.size()))
            .setPAttachments(attachments.data())
            .setSubpassCount(1)
            .setPSubpasses(&subpass)
            .setDependencyCount(1)
            .setPDependencies(&dependency);

    render_pass = vk::raii::RenderPass(device, render_pass_info);
  }

  void create_frame_buffers() {
    if (app_info.dynamic_rendering_supported) {
      // No render pass needed with dynamic rendering
      std::cout << "Using dynamic rendering, skipping framebuffer creation\n";
      return;
    }

    std::cout << "Creating traditional framebuffers\n";

    swap_chain_frame_buffers.clear();

    for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
      std::array<vk::ImageView, 3> attachmens = {
          *color_image_view, *depth_image_view, *swap_chain_image_views[i]};

      auto framebuffer_info =
          vk::FramebufferCreateInfo()
              .setRenderPass(*render_pass)
              .setAttachmentCount(static_cast<uint32_t>(attachmens.size()))
              .setPAttachments(attachmens.data())
              .setWidth(swap_chain_extent.width)
              .setHeight(swap_chain_extent.height)
              .setLayers(1);

      swap_chain_frame_buffers.emplace_back(device, framebuffer_info);
    }
  }

  void create_descriptor_set_layout() {
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
        vk::DescriptorSetLayoutBinding()
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eUniformBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eVertex),
        vk::DescriptorSetLayoutBinding()
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment)};

    auto layout_info = vk::DescriptorSetLayoutCreateInfo()
                           .setBindingCount(bindings.size())
                           .setPBindings(bindings.data());

    descriptor_set_layout = vk::raii::DescriptorSetLayout(device, layout_info);
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
                          .setFrontFace(vk::FrontFace::eCounterClockwise)
                          .setDepthBiasEnable(vk::False)
                          .setLineWidth(1.0f);

    auto multisampling = vk::PipelineMultisampleStateCreateInfo()
                             .setRasterizationSamples(msaa_samples)
                             .setSampleShadingEnable(vk::True)
                             .setMinSampleShading(0.2f);

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

    auto pipeline_layout_info = vk::PipelineLayoutCreateInfo()
                                    .setSetLayoutCount(1)
                                    .setPSetLayouts(&*descriptor_set_layout)
                                    .setPushConstantRangeCount(0);

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
                .setPColorAttachmentFormats(&swap_chain_surface_format.format)
                .setDepthAttachmentFormat(find_depth_format())};

    if (app_info.dynamic_rendering_supported) {
      std::cout << "Configuring pipeline for dynamic rendering\n";
    } else {
      std::cout << "Configurin pipeline for traditional render pass\n";
      pipeline_create_info_chain.unlink<vk::PipelineRenderingCreateInfo>();
      pipeline_create_info_chain.get<vk::GraphicsPipelineCreateInfo>()
          .setRenderPass(*render_pass);
    }

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

  void create_color_resources() {
    auto color_format = swap_chain_surface_format.format;

    std::tie(color_image, color_image_memory) =
        create_image(swap_chain_extent.width, swap_chain_extent.height, 1,
                     msaa_samples, color_format, vk::ImageTiling::eOptimal,
                     vk::ImageUsageFlagBits::eColorAttachment,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    color_image_view = create_image_view(color_image, color_format,
                                         vk::ImageAspectFlagBits::eColor, 1);
  }

  void create_depth_resources() {
    auto depth_format = find_depth_format();

    std::tie(depth_image, depth_image_memory) =
        create_image(swap_chain_extent.width, swap_chain_extent.height, 1,
                     msaa_samples, depth_format, vk::ImageTiling::eOptimal,
                     vk::ImageUsageFlagBits::eDepthStencilAttachment,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
    depth_image_view = create_image_view(depth_image, depth_format,
                                         vk::ImageAspectFlagBits::eDepth, 1);
  }
  // Available formats: eLinear || eOptimal
  vk::Format find_supported_format(const std::vector<vk::Format> &candidates,
                                   vk::ImageTiling tiling,
                                   vk::FormatFeatureFlags features) const {
    for (const auto format : candidates) {
      const auto props = physical_device.getFormatProperties(format);
      if (((tiling == vk::ImageTiling::eLinear) &&
           (props.linearTilingFeatures & features) == features) ||
          (tiling == vk::ImageTiling::eOptimal) &&
              (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }
    throw std::runtime_error("failed to find supported format!");
  }

  vk::Format find_depth_format() const {
    return find_supported_format(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
         vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

  void create_texture_image() {
    int text_width, text_height, text_channels;
    // Load Torre Archirafi pixel art
    stbi_uc *pixels = stbi_load(TEXTURE_PATH.c_str(), &text_width, &text_height,
                                &text_channels, STBI_rgb_alpha);

    vk::DeviceSize image_size = text_width * text_height * 4;

    mip_levels = static_cast<uint32_t>(
                     std::floor(std::log2(std::max(text_width, text_height)))) +
                 1;

    if (!pixels) {
      throw std::runtime_error("failed to load a image texture!");
    }

    auto [staging_buffer, staging_buffer_memory] =
        create_buffer(image_size, vk::BufferUsageFlagBits::eTransferSrc,
                      vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data = staging_buffer_memory.mapMemory(0, image_size);
    memcpy(data, pixels, image_size);
    staging_buffer_memory.unmapMemory();

    stbi_image_free(pixels);

    std::tie(texture_image, texture_image_memory) = create_image(
        text_width, text_height, mip_levels, vk::SampleCountFlagBits::e1,
        vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    transition_image_layout(texture_image, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eTransferDstOptimal, mip_levels);
    copy_buffer_to_image(staging_buffer, texture_image,
                         static_cast<uint32_t>(text_width),
                         static_cast<uint32_t>(text_height));
    generate_mipmaps(texture_image, vk::Format::eR8G8B8A8Srgb, text_width,
                     text_height, mip_levels);
  }

  void generate_mipmaps(vk::raii::Image &image, vk::Format image_format,
                        uint32_t text_w, uint32_t text_h, uint32_t mip_levels) {

    auto format_properties = physical_device.getFormatProperties(image_format);

    if (!(format_properties.optimalTilingFeatures &
          vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
      throw std::runtime_error(
          "texture image format does not support linear blitting!");
    }

    std::unique_ptr<vk::raii::CommandBuffer> command_buffer =
        begin_single_time_commands();

    auto barrier = vk::ImageMemoryBarrier()
                       .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                       .setDstAccessMask(vk::AccessFlagBits::eTransferRead)
                       .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
                       .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
                       .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
                       .setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
                       .setImage(image);

    int32_t mip_w = text_w;
    int32_t mip_h = text_h;

    for (uint32_t i = 1; i < mip_levels; i++) {
      barrier.subresourceRange.setBaseMipLevel(i - 1);
      barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
          .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
          .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
          .setDstAccessMask(vk::AccessFlagBits::eTransferRead);

      command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eTransfer, {},
                                      {}, {}, barrier);

      vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dst_offsets;
      offsets[0] = vk::Offset3D(0, 0, 0);
      offsets[1] = vk::Offset3D(mip_w, mip_h, 1);
      dst_offsets[0] = vk::Offset3D(0, 0, 0);
      dst_offsets[1] =
          vk::Offset3D(mip_w > 1 ? mip_w / 2 : 1, mip_h > 1 ? mip_h / 2 : 1, 1);

      auto blit = vk::ImageBlit()
                      .setSrcSubresource(
                          vk::ImageSubresourceLayers()
                              .setAspectMask(vk::ImageAspectFlagBits::eColor)
                              .setMipLevel(i - 1)
                              .setLayerCount(1))
                      .setSrcOffsets(offsets)
                      .setDstSubresource(
                          vk::ImageSubresourceLayers()
                              .setAspectMask(vk::ImageAspectFlagBits::eColor)
                              .setMipLevel(i)
                              .setLayerCount(1))
                      .setDstOffsets(dst_offsets);

      command_buffer->blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
                                image, vk::ImageLayout::eTransferDstOptimal,
                                blit, vk::Filter::eLinear);

      barrier.setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
          .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
          .setSrcAccessMask(vk::AccessFlagBits::eTransferRead)
          .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

      command_buffer->pipelineBarrier(
          vk::PipelineStageFlagBits::eTransfer,
          vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

      if (1 < mip_w)
        mip_w /= 2;
      if (1 < mip_h)
        mip_h /= 2;
    }
    barrier.subresourceRange.baseMipLevel = mip_levels - 1;
    barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
        .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

    command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eFragmentShader,
                                    {}, {}, {}, barrier);

    end_single_time_commands(*command_buffer);
  }

  vk::SampleCountFlagBits get_max_usable_sample_count() {
    auto physical_device_props = physical_device.getProperties();

    auto counts = physical_device_props.limits.framebufferColorSampleCounts &
                  physical_device_props.limits.framebufferDepthSampleCounts;

    if (counts & vk::SampleCountFlagBits::e64)
      return vk::SampleCountFlagBits::e64;
    if (counts & vk::SampleCountFlagBits::e32)
      return vk::SampleCountFlagBits::e32;
    if (counts & vk::SampleCountFlagBits::e16)
      return vk::SampleCountFlagBits::e16;
    if (counts & vk::SampleCountFlagBits::e8)
      return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4)
      return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2)
      return vk::SampleCountFlagBits::e2;

    return vk::SampleCountFlagBits::e1;
  }

  void create_texture_image_view() {
    texture_image_view =
        create_image_view(texture_image, vk::Format::eR8G8B8A8Srgb,
                          vk::ImageAspectFlagBits::eColor, mip_levels);
  }

  void create_texture_sampler() {
    auto properties = physical_device.getProperties();
    auto sampler_info =
        vk::SamplerCreateInfo()
            .setMagFilter(vk::Filter::eLinear)
            .setMinFilter(vk::Filter::eLinear)
            .setMipmapMode(vk::SamplerMipmapMode::eLinear)
            .setAddressModeU(vk::SamplerAddressMode::eRepeat)
            .setAddressModeV(vk::SamplerAddressMode::eRepeat)
            .setAddressModeW(vk::SamplerAddressMode::eRepeat)
            .setMipLodBias(0.0f)
            .setAnisotropyEnable(vk::True)
            .setMaxAnisotropy(properties.limits.maxSamplerAnisotropy)
            .setCompareEnable(vk::False)
            .setCompareOp(vk::CompareOp::eAlways)
            .setMinLod(0.0f)
            .setMaxLod(vk::LodClampNone);

    texture_sampler = vk::raii::Sampler(device, sampler_info);
  }

  vk::raii::ImageView create_image_view(vk::Image const &image,
                                        vk::Format format,
                                        vk::ImageAspectFlagBits aspect_mask,
                                        uint32_t mip_levels) const {
    auto view_info = vk::ImageViewCreateInfo()
                         .setFormat(format)
                         .setImage(image)
                         .setViewType(vk::ImageViewType::e2D)
                         .setSubresourceRange(vk::ImageSubresourceRange()
                                                  .setAspectMask(aspect_mask)
                                                  .setBaseArrayLayer(0)
                                                  .setBaseMipLevel(0)
                                                  .setLayerCount(1)
                                                  .setLevelCount(mip_levels));

    return vk::raii::ImageView(device, view_info);
  }

  std::pair<vk::raii::Image, vk::raii::DeviceMemory>
  create_image(uint32_t width, uint32_t height, uint32_t mip_levels,
               vk::SampleCountFlagBits num_samples, vk::Format format,
               vk::ImageTiling tiling, vk::ImageUsageFlags usage,
               vk::MemoryPropertyFlags properties) {
    auto image_info = vk::ImageCreateInfo()
                          .setImageType(vk::ImageType::e2D)
                          .setFormat(format)
                          .setExtent({width, height, 1})
                          .setMipLevels(mip_levels)
                          .setArrayLayers(1)
                          .setSamples(num_samples)
                          .setTiling(tiling)
                          .setUsage(usage)
                          .setSharingMode(vk::SharingMode::eExclusive)
                          .setInitialLayout(vk::ImageLayout::eUndefined);

    auto image = vk::raii::Image(device, image_info);

    auto mem_req = image.getMemoryRequirements();
    auto alloc_info = vk::MemoryAllocateInfo()
                          .setAllocationSize(mem_req.size)
                          .setMemoryTypeIndex(find_memory_type(
                              mem_req.memoryTypeBits, properties));
    auto image_memory = vk::raii::DeviceMemory(device, alloc_info);
    image.bindMemory(image_memory, 0);

    return {std::move(image), std::move(image_memory)};
  }

  void transition_image_layout(const vk::raii::Image &image,
                               vk::ImageLayout old_layout,
                               vk::ImageLayout new_layout,
                               uint32_t mip_levels) {
    const auto command_buffer = begin_single_time_commands();

    if (app_info.synchronization2_supported) {
      auto barrier =
          vk::ImageMemoryBarrier2()
              .setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
              .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
              .setOldLayout(old_layout)
              .setNewLayout(new_layout)
              .setImage(image)
              .setSubresourceRange(
                  vk::ImageSubresourceRange()
                      .setAspectMask(vk::ImageAspectFlagBits::eColor)
                      .setBaseArrayLayer(0)
                      .setBaseMipLevel(0)
                      .setLevelCount(mip_levels)
                      .setLayerCount(1));

      if (old_layout == vk::ImageLayout::eUndefined &&
          new_layout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.setSrcAccessMask(vk::AccessFlagBits2::eNone)
            .setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
            .setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
            .setDstStageMask(vk::PipelineStageFlagBits2::eTransfer);
      } else if (old_layout == vk::ImageLayout::eTransferDstOptimal &&
                 new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits2::eShaderRead)
            .setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
            .setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader);
      } else
        throw std::invalid_argument("unsupported layer transition!");

      auto dependency_info = vk::DependencyInfo()
                                 .setImageMemoryBarrierCount(1)
                                 .setPImageMemoryBarriers(&barrier);
      command_buffer->pipelineBarrier2(dependency_info);
    } else {
      // Use traditional synchronization API
      auto barrier = vk::ImageMemoryBarrier()
                         .setOldLayout(old_layout)
                         .setNewLayout(new_layout)
                         .setImage(image)
                         .setSubresourceRange(
                             vk::ImageSubresourceRange()
                                 .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                 .setBaseArrayLayer(0)
                                 .setBaseMipLevel(0)
                                 .setLevelCount(mip_levels)
                                 .setLayerCount(1));

      vk::PipelineStageFlags source_stage;
      vk::PipelineStageFlags destination_stage;

      if (old_layout == vk::ImageLayout::eUndefined &&
          new_layout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.setSrcAccessMask({}).setDstAccessMask(
            vk::AccessFlagBits::eTransferWrite);

        source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
        destination_stage = vk::PipelineStageFlagBits::eTransfer;
      } else if (old_layout == vk::ImageLayout::eTransferDstOptimal &&
                 new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

        source_stage = vk::PipelineStageFlagBits::eTransfer;
        destination_stage = vk::PipelineStageFlagBits::eFragmentShader;
      } else
        throw std::invalid_argument("unsupported layer transition!");

      command_buffer->pipelineBarrier(source_stage, destination_stage, {}, {},
                                      {}, barrier);
    }
    end_single_time_commands(*command_buffer);
  }

  void copy_buffer_to_image(const vk::raii::Buffer &buffer,
                            vk::raii::Image &image, uint32_t width,
                            uint32_t height) {

    std::unique_ptr<vk::raii::CommandBuffer> command_buffer =
        begin_single_time_commands();

    auto region = vk::BufferImageCopy()
                      .setBufferOffset(0)
                      .setBufferRowLength(0)
                      .setBufferImageHeight(0)
                      .setImageSubresource(
                          vk::ImageSubresourceLayers()
                              .setLayerCount(1)
                              .setMipLevel(0)
                              .setAspectMask(vk::ImageAspectFlagBits::eColor)
                              .setBaseArrayLayer(0))
                      .setImageOffset({0, 0, 0})
                      .setImageExtent({width, height, 1});
    command_buffer->copyBufferToImage(
        buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
    end_single_time_commands(*command_buffer);
  }

  void load_model() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          MODEL_PATH.c_str())) {
      throw std::runtime_error(warn + err);
    }

    std::unordered_map<Vertex, uint32_t> unique_vertices;

    for (const auto &shape : shapes) {
      for (const auto &index : shape.mesh.indices) {
        Vertex vertex{};

        vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
                      attrib.vertices[3 * index.vertex_index + 1],
                      attrib.vertices[3 * index.vertex_index + 2]};

        if (index.texcoord_index >= 0) {
          vertex.text_coord = {
              attrib.texcoords[2 * index.texcoord_index + 0],
              1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
          };
        } else {
          vertex.text_coord = {0.0f, 0.0f};
        }

        vertex.color = {1.0f, 1.0f, 1.0f};

        auto [it, inserted] = unique_vertices.insert(
            {vertex, static_cast<uint32_t>(vertices.size())});
        if (inserted) {
          vertices.push_back(vertex);
        }
        indices.push_back(it->second);
      }
    }
  }

  void create_vertex_buffer() {
    vk::DeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

    auto [stagin_buffer, stagin_buffer_memory] =
        create_buffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc,
                      vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data_stagin = stagin_buffer_memory.mapMemory(0, buffer_size);
    memcpy(data_stagin, vertices.data(), buffer_size);
    stagin_buffer_memory.unmapMemory();

    std::tie(vertex_buffer, vertex_buffer_memory) =
        create_buffer(buffer_size,
                      vk::BufferUsageFlagBits::eVertexBuffer |
                          vk::BufferUsageFlagBits::eTransferDst,
                      vk::MemoryPropertyFlagBits::eDeviceLocal);

    copy_buffer(stagin_buffer, vertex_buffer, buffer_size);
  }

  void create_index_buffer() {
    vk::DeviceSize buffer_size = sizeof(indices[0]) * indices.size();

    auto [stagin_buffer, stagin_buffer_memory] =
        create_buffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc,
                      vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data = stagin_buffer_memory.mapMemory(0, buffer_size);
    memcpy(data, indices.data(), (size_t)buffer_size);
    stagin_buffer_memory.unmapMemory();

    std::tie(index_buffer, index_buffer_memory) =
        create_buffer(buffer_size,
                      vk::BufferUsageFlagBits::eIndexBuffer |
                          vk::BufferUsageFlagBits::eTransferDst,
                      vk::MemoryPropertyFlagBits::eDeviceLocal);

    copy_buffer(stagin_buffer, index_buffer, buffer_size);
  }

  std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>
  create_buffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                vk::MemoryPropertyFlags properties) {
    auto buffer_info =
        vk::BufferCreateInfo().setSize(size).setUsage(usage).setSharingMode(
            vk::SharingMode::eExclusive);

    auto buffer = vk::raii::Buffer(device, buffer_info);

    auto mem_requirements = buffer.getMemoryRequirements();
    auto mem_allocate_info =
        vk::MemoryAllocateInfo()
            .setAllocationSize(mem_requirements.size)
            .setMemoryTypeIndex(
                find_memory_type(mem_requirements.memoryTypeBits, properties));

    auto buffer_memory = vk::raii::DeviceMemory(device, mem_allocate_info);
    buffer.bindMemory(buffer_memory, 0);

    return {std::move(buffer), std::move(buffer_memory)};
  }

  std::unique_ptr<vk::raii::CommandBuffer> begin_single_time_commands() {
    auto alloc_info = vk::CommandBufferAllocateInfo()
                          .setCommandPool(command_pool)
                          .setLevel(vk::CommandBufferLevel::ePrimary)
                          .setCommandBufferCount(1);

    auto command_buffer = std::make_unique<vk::raii::CommandBuffer>(
        std::move(vk::raii::CommandBuffers(device, alloc_info).front()));
    auto begin_info = vk::CommandBufferBeginInfo().setFlags(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    command_buffer->begin(begin_info);

    return command_buffer;
  }

  void end_single_time_commands(const vk::raii::CommandBuffer &command_buffer) {
    command_buffer.end();

    auto submit_info =
        vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(
            &*command_buffer);

    queue.submit(submit_info);
    queue.waitIdle();
  }

  void copy_buffer(vk::raii::Buffer &src_buffer, vk::raii::Buffer &dst_buffer,
                   vk::DeviceSize size) {
    auto alloc_info = vk::CommandBufferAllocateInfo()
                          .setCommandPool(command_pool)
                          .setLevel(vk::CommandBufferLevel::ePrimary)
                          .setCommandBufferCount(1);

    auto command_copy_buffer =
        std::move(device.allocateCommandBuffers(alloc_info).front());
    command_copy_buffer.begin(vk::CommandBufferBeginInfo().setFlags(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    command_copy_buffer.copyBuffer(*src_buffer, *dst_buffer,
                                   vk::BufferCopy().setSize(size));
    command_copy_buffer.end();
    queue.submit(vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(
                     &*command_copy_buffer),
                 nullptr);
    queue.waitIdle();
  }

  void create_uniform_buffers() {
    uniform_buffers.clear();
    uniform_buffers_memory.clear();
    uniform_buffers_mapped.clear();

    for (int size = 0; size < MAX_FRAMES_IN_FLIGHT; size++) {
      vk::DeviceSize buffer_size = sizeof(UniformBufferObject);
      auto [buffer, buffer_memory] =
          create_buffer(buffer_size, vk::BufferUsageFlagBits::eUniformBuffer,
                        vk::MemoryPropertyFlagBits::eHostVisible |
                            vk::MemoryPropertyFlagBits::eHostCoherent);
      uniform_buffers.emplace_back(std::move(buffer));
      uniform_buffers_memory.emplace_back(std::move(buffer_memory));
      uniform_buffers_mapped.emplace_back(
          uniform_buffers_memory.back().mapMemory(0, buffer_size));
    }
  }

  void create_descriptor_pool() {
    std::array<vk::DescriptorPoolSize, 2> pool_size{
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eUniformBuffer)
            .setDescriptorCount(MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(MAX_FRAMES_IN_FLIGHT)};
    auto pool_info =
        vk::DescriptorPoolCreateInfo()
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
            .setMaxSets(MAX_FRAMES_IN_FLIGHT)
            .setPoolSizeCount(static_cast<uint32_t>(pool_size.size()))
            .setPPoolSizes(pool_size.data());

    descriptor_pool = vk::raii::DescriptorPool(device, pool_info);
  }

  void create_descriptor_sets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                 descriptor_set_layout);
    auto alloc_info =
        vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(descriptor_pool)
            .setDescriptorSetCount(static_cast<uint32_t>(layouts.size()))
            .setPSetLayouts(layouts.data());

    descriptor_sets.clear();
    descriptor_sets = device.allocateDescriptorSets(alloc_info);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      auto buffer_info = vk::DescriptorBufferInfo()
                             .setBuffer(uniform_buffers[i])
                             .setOffset(0)
                             .setRange(sizeof(UniformBufferObject));
      auto image_info =
          vk::DescriptorImageInfo()
              .setSampler(texture_sampler)
              .setImageView(texture_image_view)
              .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

      std::array<vk::WriteDescriptorSet, 2> descriptor_writes = {
          vk::WriteDescriptorSet()
              .setDstSet(descriptor_sets[i])
              .setDstBinding(0)
              .setDstArrayElement(0)
              .setDescriptorCount(1)
              .setDescriptorType(vk::DescriptorType::eUniformBuffer)
              .setPBufferInfo(&buffer_info),
          vk::WriteDescriptorSet()
              .setDstSet(descriptor_sets[i])
              .setDstBinding(1)
              .setDstArrayElement(0)
              .setDescriptorCount(1)
              .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
              .setPImageInfo(&image_info)};

      device.updateDescriptorSets(descriptor_writes, {});
    }
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
    command_buffers.clear();
    auto alloc_info = vk::CommandBufferAllocateInfo()
                          .setCommandPool(command_pool)
                          .setLevel(vk::CommandBufferLevel::ePrimary)
                          .setCommandBufferCount(MAX_FRAMES_IN_FLIGHT);
    command_buffers = vk::raii::CommandBuffers(device, alloc_info);
  }

  void record_command_buffer(uint32_t image_index) {
    auto &command_buffer = command_buffers[frame_index];
    command_buffer.begin({});

    auto clear_color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    auto clear_depth = vk::ClearDepthStencilValue(1.0f, 0);

    std::array<vk::ClearValue, 2> clear_values = {clear_color, clear_depth};

    if (app_info.dynamic_rendering_supported) {
      if (app_info.synchronization2_supported) {
        assert(color_image != nullptr);
        assert(depth_image != nullptr);
        auto color_barrier =
            vk::ImageMemoryBarrier2()
                .setSrcStageMask(
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
                .setDstStageMask(
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setImage(*color_image)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseMipLevel(0)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1));

        auto depth_barrier =
            vk::ImageMemoryBarrier2()
                .setSrcStageMask(
                    vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                    vk::PipelineStageFlagBits2::eLateFragmentTests)
                .setSrcAccessMask(
                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
                .setDstStageMask(
                    vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                    vk::PipelineStageFlagBits2::eLateFragmentTests)
                .setDstAccessMask(
                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
                .setImage(*depth_image)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eDepth)
                        .setBaseMipLevel(0)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1));

        auto swapchain_barrier =
            vk::ImageMemoryBarrier2()
                .setSrcStageMask(
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                .setDstStageMask(
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setImage(swap_chain_images[image_index])
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseMipLevel(0)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1));
        std::array<vk::ImageMemoryBarrier2, 3> barriers = {
            color_barrier, depth_barrier, swapchain_barrier};
        auto dependency_info = vk::DependencyInfo()
                                   .setImageMemoryBarrierCount(
                                       static_cast<uint32_t>(barriers.size()))
                                   .setPImageMemoryBarriers(barriers.data());
        command_buffer.pipelineBarrier2(dependency_info);
      } else {
        auto color_barrier =
            vk::ImageMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eNone)
                .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setImage(*color_image)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseArrayLayer(0)
                        .setBaseMipLevel(0)
                        .setLayerCount(1)
                        .setLevelCount(1));

        auto depth_barrier =
            vk::ImageMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eNone)
                .setDstAccessMask(
                    vk::AccessFlagBits::eDepthStencilAttachmentWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setImage(*depth_image)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseArrayLayer(0)
                        .setBaseMipLevel(0)
                        .setLayerCount(1)
                        .setLevelCount(1));

        auto swapchain_barrier =
            vk::ImageMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eNone)
                .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setImage(swap_chain_images[image_index])
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseArrayLayer(0)
                        .setBaseMipLevel(0)
                        .setLayerCount(1)
                        .setLevelCount(1));

        std::array<vk::ImageMemoryBarrier, 3> barriers = {
            color_barrier, depth_barrier, swapchain_barrier};

        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eColorAttachmentOutput |
                vk::PipelineStageFlagBits::eEarlyFragmentTests,
            vk::DependencyFlagBits::eByRegion, {}, {}, barriers);
      }

      // Setup rendering attachments
      auto color_attachment_info =
          vk::RenderingAttachmentInfo()
              .setImageView(*color_image_view)
              .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
              .setResolveMode(vk::ResolveModeFlagBits::eAverage)
              .setResolveImageView(*swap_chain_image_views[image_index])
              .setResolveImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
              .setLoadOp(vk::AttachmentLoadOp::eClear)
              .setStoreOp(vk::AttachmentStoreOp::eStore)
              .setClearValue(clear_color);
      auto depth_attachment_info =
          vk::RenderingAttachmentInfo()
              .setImageView(*depth_image_view)
              .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
              .setLoadOp(vk::AttachmentLoadOp::eClear)
              .setStoreOp(vk::AttachmentStoreOp::eDontCare)
              .setClearValue(clear_depth);

      auto rendering_info =
          vk::RenderingInfo()
              .setRenderArea(
                  vk::Rect2D().setOffset({0, 0}).setExtent(swap_chain_extent))
              .setLayerCount(1)
              .setColorAttachmentCount(1)
              .setPColorAttachments(&color_attachment_info)
              .setPDepthAttachment(&depth_attachment_info);

      command_buffer.beginRendering(rendering_info);
    } else {
      std::cout << "Recording command buffer with traditional render pass\n";

      auto render_pass_info =
          vk::RenderPassBeginInfo()
              .setRenderPass(*render_pass)
              .setFramebuffer(*swap_chain_frame_buffers[image_index])
              .setRenderArea(
                  vk::Rect2D().setOffset({0, 0}).setExtent(swap_chain_extent))
              .setClearValueCount(static_cast<uint32_t>(clear_values.size()))
              .setPClearValues(clear_values.data());

      command_buffer.beginRenderPass(render_pass_info,
                                     vk::SubpassContents::eInline);
    }

    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                *graphics_pipeline);
    command_buffer.setViewport(
        0,
        vk::Viewport(0.0f, 0.0f, static_cast<float>(swap_chain_extent.width),
                     static_cast<float>(swap_chain_extent.height), 0.0f, 1.0f));
    command_buffer.setScissor(
        0, vk::Rect2D(vk::Offset2D(0, 0), swap_chain_extent));
    command_buffer.bindVertexBuffers(0, *vertex_buffer, {0});
    command_buffer.bindIndexBuffer(*index_buffer, 0, vk::IndexType::eUint32);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                      pipeline_layout, 0,
                                      *descriptor_sets[frame_index], nullptr);
    command_buffer.drawIndexed(indices.size(), 1, 0, 0, 0);

    if (app_info.dynamic_rendering_supported) {
      command_buffer.endRendering();

      if (app_info.synchronization2_supported) {
        auto barrier =
            vk::ImageMemoryBarrier2()
                .setSrcStageMask(
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
                .setDstStageMask(vk::PipelineStageFlagBits2::eBottomOfPipe)
                .setDstAccessMask(vk::AccessFlagBits2::eNone)
                .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
                .setImage(swap_chain_images[image_index])
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseArrayLayer(0)
                        .setBaseMipLevel(0)
                        .setLayerCount(1)
                        .setLevelCount(1));

        auto dependency_info = vk::DependencyInfo()
                                   .setImageMemoryBarrierCount(1)
                                   .setPImageMemoryBarriers(&barrier);
        command_buffer.pipelineBarrier2(dependency_info);
      } else {
        auto barrier =
            vk::ImageMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
                .setDstAccessMask(vk::AccessFlagBits::eNone)
                .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setImage(swap_chain_images[image_index])
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseArrayLayer(0)
                        .setBaseMipLevel(0)
                        .setLayerCount(1)
                        .setLevelCount(1));

        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eBottomOfPipe,
            vk::DependencyFlagBits::eByRegion, {}, {}, {barrier});
      }
    } else {
      command_buffer.endRenderPass();
    }

    command_buffer.end();
  }

  void create_sync_objects() {
    assert(present_complete_semaphores.empty() &&
           render_finished_semaphores.empty() && in_flight_fences.empty());

    if (app_info.timeline_sempaphore_supported) {
      std::cout << "Creating timeline semaphores\n";
      auto timeline_create_info =
          vk::SemaphoreTypeCreateInfo()
              .setSemaphoreType(vk::SemaphoreType::eTimeline)
              .setInitialValue(0);

      auto semaphore_info =
          vk::SemaphoreCreateInfo().setPNext(&timeline_create_info);

      time_line_semaphore = vk::raii::Semaphore(device, semaphore_info);
    }

    for (size_t i = 0; i < swap_chain_images.size(); i++) {
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

  void update_uniform_buffer(uint32_t current_image) const {
    static auto start_time = std::chrono::high_resolution_clock::now();

    auto current_time = std::chrono::high_resolution_clock::now();
    float time =
        std::chrono::duration<float>(current_time - start_time).count();

    std::cout << time;

    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj =
        glm::perspective(glm::radians(45.0f),
                         static_cast<float>(swap_chain_extent.width) /
                             static_cast<float>(swap_chain_extent.height),
                         0.1f, 10.0f);
    ubo.proj[1][1] *= -1;
    memcpy(uniform_buffers_mapped[current_image], &ubo, sizeof(ubo));
  }

  void draw_frame() {
    // Note: inFlightFences, presentCompleteSemaphores, and commandBuffers are
    // indexed by frameIndex,
    // while renderFinishedSemaphores is indexed by imageIndex
    auto fence_result = device.waitForFences(*in_flight_fences[frame_index],
                                             vk::True, UINT64_MAX);
    if (fence_result != vk::Result::eSuccess)
      throw std::runtime_error("failed to wait for fence!");

    auto [result, image_index] = swap_chain.acquireNextImage(
        UINT64_MAX, *present_complete_semaphores[frame_index], nullptr);

    // Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined,
    // eErrorOutOfDateKHR can be checked as a result here and does not need to
    // be caught by an exception.
    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreate_swap_chain();
      return;
    }
    // On other success codes than eSuccess and eSuboptimalKHR we just throw an
    // exception. On any error code, aquireNextImage already threw an exception.
    if (result != vk::Result::eSuccess &&
        result != vk::Result::eSuboptimalKHR) {
      assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
      throw std::runtime_error("failed to acquire swap chain image");
    }
    update_uniform_buffer(frame_index);
    // Only reset the fence if we are submitting work
    device.resetFences(*in_flight_fences[frame_index]);

    command_buffers[frame_index].reset();
    record_command_buffer(image_index);

    if (app_info.timeline_sempaphore_supported) {
      uint64_t wait_value = time_line_value;
      uint64_t signal_value = ++time_line_value;

      auto timeline_info = vk::TimelineSemaphoreSubmitInfo()
                               .setWaitSemaphoreValueCount(0)
                               .setSignalSemaphoreValueCount(1)
                               .setPSignalSemaphoreValues(&signal_value);

      std::array<vk::Semaphore, 2> wait_semaphores = {
          *present_complete_semaphores[frame_index], *time_line_semaphore};
      std::array<vk::PipelineStageFlags, 2> wait_stages = {
          vk::PipelineStageFlagBits::eColorAttachmentOutput,
          vk::PipelineStageFlagBits::eVertexInput};
      std::array<uint64_t, 2> wait_values = {0, wait_value};

          std::array<vk::Semaphore, 2> signal_semaphores = {
            *render_finished_semaphores[frame_index], *time_line_semaphore};
      std::array<uint64_t, 2> signal_values = {0, signal_value};

      timeline_info.setWaitSemaphoreValueCount(2)
          .setPWaitSemaphoreValues(wait_values.data())
          .setSignalSemaphoreValueCount(2)
          .setPSignalSemaphoreValues(signal_values.data());

      auto submit_info = vk::SubmitInfo()
                             .setPNext(&timeline_info)
                             .setWaitSemaphoreCount(2)
                             .setPWaitSemaphores(wait_semaphores.data())
                             .setPWaitDstStageMask(wait_stages.data())
                             .setCommandBufferCount(1)
                             .setPCommandBuffers(&*command_buffers[frame_index])
                             .setSignalSemaphoreCount(2)
                             .setPSignalSemaphores(signal_semaphores.data());

      queue.submit(submit_info, *in_flight_fences[frame_index]);
      std::cout << "Prima fase\n";
    } else {
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
    }

    const auto present_info_khr =
        vk::PresentInfoKHR()
            .setWaitSemaphoreCount(1)
            .setPWaitSemaphores(&*render_finished_semaphores[frame_index])
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
    frame_index = (frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
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

  const std::vector<const char *> get_required_instance_extensions() const {
    uint32_t glfw_ext_count = 0;
    auto glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    std::vector extensions(glfw_exts, glfw_exts + glfw_ext_count);

    // Check if the debug utils extesion is available
    std::vector<vk::ExtensionProperties> props =
        context.enumerateInstanceExtensionProperties();
    bool debug_utils_available = std::any_of(
        props.begin(), props.end(), [](vk::ExtensionProperties const &ep) {
          return strcmp(ep.extensionName, vk::EXTDebugUtilsExtensionName) == 0;
        });

    // Always include the debug utils extension if available
    // This allows validation layer to be enabled via vulkanconfig
    if (debug_utils_available)
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    else
      std::cout << "VK_EXT_debug_utils extension not available. Validation "
                   "layers may not work."
                << std::endl;

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
    HelloTriangleApplication app;
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
