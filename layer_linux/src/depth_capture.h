#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace dlssnr {

// Finds the game's main depth buffer so the model can see the scene's shape.
//
// A present-time layer only sees the finished picture, so we watch the game record its frame
// instead: every depth attachment it renders into, and how many vertices each one receives. At
// present the busiest depth buffer whose shape matches the swapchain is the scene's; shadow maps and
// other offscreen passes are square or small and fall out on the shape test. This is the heuristic
// ReShade's generic depth add-on uses, reduced to what an emulator's frame needs.
class DepthCapture {
public:
    // DLSSNR_DEPTH=1 switches the capture on. Off, the layer hooks nothing it did not hook before.
    static bool Enabled();
    static bool IsDepthFormat(VkFormat format);

    void OnImageCreated(VkImage image, const VkImageCreateInfo& info);
    void OnImageDestroyed(VkImage image);
    void OnViewCreated(VkImageView view, VkImage image);
    void OnViewDestroyed(VkImageView view);
    void OnFramebufferCreated(VkFramebuffer framebuffer, const VkFramebufferCreateInfo& info);
    void OnFramebufferDestroyed(VkFramebuffer framebuffer);
    void OnRenderPassCreated(VkRenderPass renderPass, int depthAttachment, VkImageLayout finalLayout);
    void OnRenderPassDestroyed(VkRenderPass renderPass);

    void BeginRendering(VkCommandBuffer cb, VkImageView depthView, VkImageLayout layout);
    void BeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo& info);
    void Draw(VkCommandBuffer cb, uint64_t vertices);
    void EndPass(VkCommandBuffer cb);
    void Barrier(VkImage image, VkImageLayout newLayout);

    struct Pick {
        VkImage image = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0, height = 0;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };
    // The scene's depth buffer for the frame being presented. Starts the count for the next frame.
    bool PickAndReset(uint32_t swapWidth, uint32_t swapHeight, Pick* pick);

private:
    struct Image {
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0, height = 0;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        uint64_t vertices = 0;
    };
    struct Pass {
        VkImage depth = VK_NULL_HANDLE;
        VkImageLayout layoutAfter = VK_IMAGE_LAYOUT_UNDEFINED;
        uint64_t vertices = 0;
    };
    struct RenderPass {
        int depthAttachment = -1;
        VkImageLayout finalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    void BeginLocked(VkCommandBuffer cb, VkImageView depthView, VkImageLayout layoutAfter);

    std::mutex _lock;
    std::unordered_map<VkImage, Image> _images;
    std::unordered_map<VkImageView, VkImage> _views;
    std::unordered_map<VkFramebuffer, std::vector<VkImageView>> _framebuffers;
    std::unordered_map<VkRenderPass, RenderPass> _renderPasses;
    std::unordered_map<VkCommandBuffer, Pass> _passes;
};

}  // namespace dlssnr
