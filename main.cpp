#include "vulkan/vulkan.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_to_string.hpp>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#if defined (__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
# include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
//#include <GLFW/glfw3native.h>


constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector<char const*> validation_layers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enable_validation_layers = false;
#else
constexpr bool enable_validation_layers = true;
#endif

class HelloTriangleApplication {
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
        vk::raii::SurfaceKHR surface = nullptr;
        vk::raii::DebugUtilsMessengerEXT debug_messenger = nullptr;
        vk::raii::PhysicalDevice physical_device = nullptr;
        vk::PhysicalDeviceFeatures physical_device_features;
        vk::raii::Device device = nullptr;
        vk::raii::Queue graphics_queue = nullptr;
        std::vector<const char*> required_device_extensions = {vk::KHRSwapchainExtensionName};

        void init_window() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

            window = glfwCreateWindow(WIDTH, HEIGHT, "Vukan", nullptr, nullptr);

        }

        void init_vulkan(){
            createInstance();
			setup_debug_messenger();
			pick_physical_device();
			create_logical_device();
        }


        void main_loop()
        {
            while (!glfwWindowShouldClose(window))
            {
                glfwPollEvents();
            }
        }

        void cleanup() {
            glfwDestroyWindow(window);

            glfwTerminate();
        }

        void createInstance() {
            vk::ApplicationInfo const appInfo = vk::ApplicationInfo()
                .setPApplicationName("Hello Triangle")
                .setPEngineName("No Engine")
                .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
                .setApiVersion(vk::ApiVersion14);

            //Get required layers
            std::vector<char const*> required_layers;
            if (enable_validation_layers) {
                required_layers.assign(validation_layers.begin(), validation_layers.end());
            }

            auto layer_properties = context.enumerateInstanceLayerProperties();

            // Cerca in required_layers il PRIMO layer che NON viene trovato in layer_properties
            auto unsupported_layer_it = std::find_if(
                required_layers.begin(),
                required_layers.end(),
                [&layer_properties](auto const &required_layer) {
                    // Catturo layer_properties. required_layer è un singolo layer richiesto.
                    // none_of restituisce true se NESSUN layer della GPU corrisponde a required_layer.
                    return std::none_of(
                        layer_properties.begin(),
                        layer_properties.end(),
                        [required_layer](auto const &layer_property) {
                            // Confronta il singolo layer GPU con il singolo layer richiesto
                            return strcmp(layer_property.layerName, required_layer) == 0; // == 0 FONDAMENTALE!
                        }
                    );
                }
            );

            //Confronto la posizione dell'elemento trovato.
            // Se risulta uguale a required_layers.end() cioè la posizione successiva all'ultimo elemento
            // Allora tutti tutti i layer sono supportati.
            if (unsupported_layer_it != required_layers.end()) {
                throw  std::runtime_error("Required layer not supported: " + std::string(*unsupported_layer_it));
            }

            //Get  the required extensions
            auto required_exts = get_required_instance_extensions();
            auto extension_properties = context.enumerateInstanceExtensionProperties();

            auto unsupported_property_it = std::find_if(
                required_exts.begin(),required_exts.end(),
                [&extension_properties](auto const &required_extension) {
                    return std::none_of(
                        extension_properties.begin(),extension_properties.end(),
                        [required_extension](auto const& extension_property) {
                            return strcmp(extension_property.extensionName, required_extension) == 0;
                        }
                    );
                }
            );

            if (unsupported_property_it != required_exts.end())
            {
                throw std::runtime_error("Required extension not supported: " + std::string(*unsupported_property_it));
            }

            vk::InstanceCreateInfo const createInfo = vk::InstanceCreateInfo()
                .setPApplicationInfo(&appInfo)
                .setEnabledLayerCount(static_cast<uint32_t>(required_layers.size()))
                .setPpEnabledLayerNames(required_layers.data())
                .setEnabledExtensionCount(static_cast<uint32_t>(required_exts.size()))
                .setPpEnabledExtensionNames(required_exts.data());

            instance = vk::raii::Instance(context,createInfo);
        }

        bool is_device_suitable(vk::raii::PhysicalDevice const & physical_device) {
            bool supported_vulkan1_3 = physical_device.getProperties().apiVersion;

            auto queue_families = physical_device.getQueueFamilyProperties();
            bool supports_graphics = std::any_of(
                queue_families.begin(),
                queue_families.end(),
                [](vk::QueueFamilyProperties const & qfp) {
                    return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
            });
            auto available_device_extensions = physical_device.enumerateDeviceExtensionProperties();
            bool supported_all_required_extensions =
                std::all_of(required_device_extensions.begin(),required_device_extensions.end(),
                    [&available_device_extensions](auto const & required_device_extension) {
                        return std::any_of(available_device_extensions.begin(),
                            available_device_extensions.end(),
                            [required_device_extension](auto const & available_device_extension){
                                return strcmp(available_device_extension.extensionName, required_device_extension) == 0;
                            });
                    });

            auto features = physical_device.getFeatures2<
                vk::PhysicalDeviceFeatures2,
                vk::PhysicalDeviceVulkan11Features,
                vk::PhysicalDeviceVulkan13Features,
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
                >();

            bool supports_required_features = features.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

            return  supports_required_features && supported_all_required_extensions && supported_vulkan1_3 && supports_graphics;

        }
        void pick_physical_device() {
            std::vector<vk::raii::PhysicalDevice> physical_devices = instance.enumeratePhysicalDevices();
            auto const dev_iter = std::find_if(
                physical_devices.begin(),
                physical_devices.end(),
                [&](vk::raii::PhysicalDevice const & pd) {
                    return is_device_suitable(pd);
                }
            );
            if (dev_iter == physical_devices.end()) throw std::runtime_error("Failed to find GPUs with Vulkan support!");
            physical_device = *dev_iter;
        }

        void create_logical_device() {
            std::vector<vk::QueueFamilyProperties> queue_family_properties = physical_device.getQueueFamilyProperties();
            auto graphics_queue_family_proprerty = std::find_if(
                queue_family_properties.begin(),queue_family_properties.end(),
                [](vk::QueueFamilyProperties const &qfp) {
                    return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
                });
            auto graphics_index = static_cast<uint32_t>(std::distance(queue_family_properties.begin(),graphics_queue_family_proprerty));

            auto feature_chain = vk::StructureChain<
                vk::PhysicalDeviceFeatures2,
                vk::PhysicalDeviceVulkan11Features,
                vk::PhysicalDeviceVulkan13Features,
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
            > {
                vk::PhysicalDeviceFeatures2{},
                vk::PhysicalDeviceVulkan11Features().setShaderDrawParameters(true),
                vk::PhysicalDeviceVulkan13Features().setDynamicRendering(true),
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT().setExtendedDynamicState(true)
            };

            std::vector<const char*>required_device_extension = {vk::KHRSwapchainExtensionName};


            auto device_queue_create_info = vk::DeviceQueueCreateInfo()
                .setQueueFamilyIndex(graphics_index);

            auto device_create_info = vk::DeviceCreateInfo()
                .setPNext(&feature_chain.get<vk::PhysicalDeviceFeatures2>())
                .setQueueCreateInfoCount(1)
                .setPQueueCreateInfos(&device_queue_create_info)
                .setEnabledExtensionCount(static_cast<uint32_t>(required_device_extension.size()))
                .setPpEnabledExtensionNames(required_device_extension.data());

            device = vk::raii::Device(physical_device,device_create_info);
            graphics_queue = vk::raii::Queue(device,graphics_index,0);
        }

        void setup_debug_messenger() {
            if (!enable_validation_layers) return;

            vk::DebugUtilsMessageSeverityFlagsEXT severity_flags(
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
            );

            vk::DebugUtilsMessageTypeFlagsEXT message_type_flags(
                vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
            );

            vk::DebugUtilsMessengerCreateInfoEXT const debug_utils_create_info_ext = vk::DebugUtilsMessengerCreateInfoEXT()
                .setMessageSeverity(severity_flags)
                .setMessageType(message_type_flags)
                .setPfnUserCallback(&debug_callback);

            debug_messenger = instance.createDebugUtilsMessengerEXT(debug_utils_create_info_ext);

        }
        const std::vector<const char*> get_required_instance_extensions() {
            uint32_t glfw_ext_count = 0;
            auto glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

            std::vector extensions(glfw_exts,glfw_exts+glfw_ext_count);
            if (enable_validation_layers) {
                extensions.push_back(vk::EXTDebugUtilsExtensionName);
            }

            return extensions;
        }

        static VKAPI_ATTR vk::Bool32 VKAPI_CALL debug_callback(
            vk::DebugUtilsMessageSeverityFlagBitsEXT      severity,
            vk::DebugUtilsMessageTypeFlagsEXT        type,
            const vk::DebugUtilsMessengerCallbackDataEXT *pCall_back_data,
            void                                          *pUser_data
        ){
            std::cerr << "validation layer: type " << vk::to_string(type) << " msg: " << pCall_back_data->pMessage << std::endl;
            return vk::False;
        }
};

int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
