#include "depth_capture.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace dlssnr {

bool DepthCapture::Enabled() {
    static const bool on = [] {
        const char* v = getenv("DLSSNR_DEPTH");
        return v && std::strcmp(v, "1") == 0;
    }();
    return on;
}

bool DepthCapture::IsDepthFormat(VkFormat format) {
    switch (format) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return true;
        default:
            return false;
    }
}

void DepthCapture::OnImageCreated(VkImage image, const VkImageCreateInfo& info) {
    if (!IsDepthFormat(info.format) || !(info.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) return;
    std::lock_guard<std::mutex> lk(_lock);
    Image& img = _images[image];
    img.format = info.format;
    img.width = info.extent.width;
    img.height = info.extent.height;
    img.layout = info.initialLayout;
}

void DepthCapture::OnImageDestroyed(VkImage image) {
    std::lock_guard<std::mutex> lk(_lock);
    _images.erase(image);
}

void DepthCapture::OnViewCreated(VkImageView view, VkImage image) {
    std::lock_guard<std::mutex> lk(_lock);
    if (_images.count(image)) _views[view] = image;
}

void DepthCapture::OnViewDestroyed(VkImageView view) {
    std::lock_guard<std::mutex> lk(_lock);
    _views.erase(view);
}

void DepthCapture::OnFramebufferCreated(VkFramebuffer framebuffer, const VkFramebufferCreateInfo& info) {
    if (info.flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT) return;
    std::lock_guard<std::mutex> lk(_lock);
    _framebuffers[framebuffer].assign(info.pAttachments, info.pAttachments + info.attachmentCount);
}

void DepthCapture::OnFramebufferDestroyed(VkFramebuffer framebuffer) {
    std::lock_guard<std::mutex> lk(_lock);
    _framebuffers.erase(framebuffer);
}

void DepthCapture::OnRenderPassCreated(VkRenderPass renderPass, int depthAttachment, VkImageLayout finalLayout) {
    std::lock_guard<std::mutex> lk(_lock);
    _renderPasses[renderPass] = RenderPass{depthAttachment, finalLayout};
}

void DepthCapture::OnRenderPassDestroyed(VkRenderPass renderPass) {
    std::lock_guard<std::mutex> lk(_lock);
    _renderPasses.erase(renderPass);
}

void DepthCapture::BeginLocked(VkCommandBuffer cb, VkImageView depthView, VkImageLayout layoutAfter) {
    Pass& pass = _passes[cb];
    pass = Pass{};
    auto view = _views.find(depthView);
    if (view == _views.end()) return;
    pass.depth = view->second;
    pass.layoutAfter = layoutAfter;
}

void DepthCapture::BeginRendering(VkCommandBuffer cb, VkImageView depthView, VkImageLayout layout) {
    std::lock_guard<std::mutex> lk(_lock);
    BeginLocked(cb, depthView, layout);
}

void DepthCapture::BeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo& info) {
    std::lock_guard<std::mutex> lk(_lock);
    auto rp = _renderPasses.find(info.renderPass);
    if (rp == _renderPasses.end() || rp->second.depthAttachment < 0) {
        _passes[cb] = Pass{};
        return;
    }
    const uint32_t index = uint32_t(rp->second.depthAttachment);
    VkImageView view = VK_NULL_HANDLE;
    // An imageless framebuffer names its attachments only when the pass begins.
    for (auto* next = static_cast<const VkBaseInStructure*>(info.pNext); next; next = next->pNext) {
        if (next->sType != VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO) continue;
        const auto* attachments = reinterpret_cast<const VkRenderPassAttachmentBeginInfo*>(next);
        if (index < attachments->attachmentCount) view = attachments->pAttachments[index];
    }
    if (!view) {
        auto fb = _framebuffers.find(info.framebuffer);
        if (fb != _framebuffers.end() && index < fb->second.size()) view = fb->second[index];
    }
    BeginLocked(cb, view, rp->second.finalLayout);
}

void DepthCapture::Draw(VkCommandBuffer cb, uint64_t vertices) {
    std::lock_guard<std::mutex> lk(_lock);
    auto pass = _passes.find(cb);
    if (pass != _passes.end() && pass->second.depth) pass->second.vertices += vertices;
}

void DepthCapture::EndPass(VkCommandBuffer cb) {
    std::lock_guard<std::mutex> lk(_lock);
    auto pass = _passes.find(cb);
    if (pass == _passes.end()) return;
    auto img = _images.find(pass->second.depth);
    if (img != _images.end()) {
        img->second.vertices += pass->second.vertices;
        img->second.layout = pass->second.layoutAfter;
    }
    _passes.erase(pass);
}

void DepthCapture::Barrier(VkImage image, VkImageLayout newLayout) {
    std::lock_guard<std::mutex> lk(_lock);
    auto img = _images.find(image);
    if (img != _images.end()) img->second.layout = newLayout;
}

bool DepthCapture::PickAndReset(uint32_t swapWidth, uint32_t swapHeight, Pick* pick) {
    std::lock_guard<std::mutex> lk(_lock);
    const double swapAspect = double(swapWidth) / double(swapHeight);
    uint64_t best = 0;
    *pick = Pick{};
    for (auto& [image, img] : _images) {
        const uint64_t vertices = img.vertices;
        img.vertices = 0;
        if (!vertices || !img.width || !img.height) continue;
        // Internal resolution may sit above or below the swapchain; the shape is what has to agree.
        const double aspect = double(img.width) / double(img.height);
        const double scale = double(img.width) / double(swapWidth);
        if (std::fabs(aspect - swapAspect) > 0.1 || scale < 0.5 || scale > 4.0) continue;
        if (img.layout == VK_IMAGE_LAYOUT_UNDEFINED || img.layout == VK_IMAGE_LAYOUT_PREINITIALIZED) continue;
        if (vertices > best) {
            best = vertices;
            *pick = Pick{image, img.format, img.width, img.height, img.layout};
        }
    }
    return pick->image != VK_NULL_HANDLE;
}

}  // namespace dlssnr
