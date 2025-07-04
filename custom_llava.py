# llava/model/custom_llava.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, LlavaForConditionalGeneration, LlavaConfig
from typing import List, Optional, Tuple, Union
import sys
import os
import logging
logger = logging.getLogger(__name__)
superyolo_path = '/home/zty/superyolo'
if superyolo_path not in sys.path:
    sys.path.insert(0, superyolo_path)
    print(f"Added {superyolo_path} to sys.path")

# Import SuperYOLO model - requires SuperYOLO path in sys.path
try:
    # Assuming your SRyolo.py is within SuperYOLO/models/SRyolo.py
    from models.SRyolo import Model as SuperYOLOModel
    print("Successfully imported SuperYOLOModel.")
except ImportError as e:
    print(f"Error: Could not import SuperYOLOModel from {superyolo_path}. Make sure the path is correct and SuperYOLO dependencies are installed.")
    print(f"ImportError: {e}")
    # Define a dummy class or handle this error appropriately
    SuperYOLOModel = None

try:
    # Attempt to get from transformers config or assumed common values
    from transformers.models.llava.modeling_llava import image_token_index, IGNORE_INDEX
    print("Successfully imported LLaVA constants.")
except ImportError:
    # Fallback to common LLaVA constants if transformers doesn't provide them directly
    # These are standard values, but double-check against your LLaVA version if issues arise.
    IMAGE_TOKEN_INDEX = -200
    IGNORE_INDEX = -100
    print(f"Could not import LLaVA constants from transformers. Using default values: IMAGE_TOKEN_INDEX={IMAGE_TOKEN_INDEX}, IGNORE_INDEX={IGNORE_INDEX}")
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration, GenerationMixin):
    def forward(self, input_ids=None, pixel_values=None, yolo_images= None,attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, **kwargs):
        # print("DEBUG: Custom forward is called!")
        
        # 如果有图像，调用自定义的 prepare_inputs_labels_for_multimodal
        if inputs_embeds is None:
            # 仅在训练或需要从头计算嵌入时调用
            if pixel_values is not None:
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    labels=labels,
                    images=pixel_values,
                    yolo_images=yolo_images
                )
            # 如果连 pixel_values 都没有，就只处理文本（例如，纯文本输入的情况）
            else:
                inputs_embeds = self.get_input_embeddings()(input_ids)
    
        # 调用父类的 forward 方法继续处理
    # 用计算好或传递进来的 inputs_embeds 调用父类 forward
    # 注意：此时 pixel_values 必须为 None，避免父类重复处理
        return super().forward(
            input_ids=input_ids,
            pixel_values=None,  # <-- 关键：设为 None
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs
        )
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, yolo_images=None,attention_mask=None, **kwargs
    ):
        print("Warning: prepare_inputs_for_generation is running")
            # 从 kwargs 中提取我们自定义的图像参数，并从字典中移除它们
        clip_images = kwargs.pop("images", None)
        yolo_images = kwargs.pop("yolo_images", None)
    
        inputs_embeds = None
        if past_key_values is None:
        # 确保 yolo_images 参数能被接收
            kwargs['yolo_images'] = yolo_images
            if inputs_embeds is None and pixel_values is not None:
                    _, _, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
                    input_ids=input_ids,
                    position_ids=None,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    labels=None,
                    images=clip_images,
                    yolo_images=yolo_images
                )
            kwargs['inputs_embeds'] = inputs_embeds  
        else:
        # 在后续生成步骤中，inputs_embeds 已经处理过，不需要再动
            inputs_embeds = None
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs
        )
        # 清理掉原始图像，因为它们已经被编码到 inputs_embeds 中了
        # 这是为了防止修改后的 forward 方法或父类方法产生混淆
        if 'pixel_values' in model_inputs:
            model_inputs.pop('pixel_values')
        if 'yolo_images' in model_inputs:
            model_inputs.pop('yolo_images')
            
        return model_inputs

    def __init__(self, config: LlavaConfig):
        # Initialize the standard LLaVA model (LLM + CLIP)
        super().__init__(config)
            # 添加特征归一化层

        self.config.image_token_id = getattr(config, 'image_token_index', 151646)
        self.debug_features_viz = getattr(config,'debug_features_viz', False) # 从config中获取，默认为False
        self.debug_output_dir = "debug_feature_visualizations_yolo"
        if self.debug_features_viz and not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir)
            print(f"Created debug output directory: {self.debug_output_dir}")
        self.visualization_counter = 0  # 新增：样本计数器

        self.yolo_cfg_path = getattr(config, 'yolo_cfg_path', None)
        self.yolo_weights_path = getattr(config, 'yolo_weights_path', None)
        self.yolo_feature_layer_idx = getattr(config, 'yolo_feature_layer_idx', None) # e.g., 6
        self.yolo_feature_layer_channels = getattr(config, 'yolo_feature_layer_channels', None) # e.g., 256
        self.yolo_patch_size = getattr(config, 'yolo_patch_size', 14)
        self.yolo_input_size = getattr(config, 'yolo_input_size', [512, 512]) # (H, W)
        # 先把 superyolo_model / yolo_projector 都置为 None
        self.yolo_projector = None
        # Basic checks检查并修正 yolo_input_size，确保是二维且能被 32 整除
        if self.yolo_input_size is None or len(self.yolo_input_size) != 2:
             print(f"Warning: yolo_input_size not properly configured, defaulting to [512, 512].")
             self.yolo_input_size = [512, 512]
        if self.yolo_input_size[0] % 32 != 0 or self.yolo_input_size[1] % 32 != 0:
             print(f"Warning: Configured YOLO input size {self.yolo_input_size} is not divisible by 32. This may cause issues.")

        # 2. Load SuperYOLO Model
        self.superyolo_model = None

    def init_superyolo(self, map_location="cpu"):
        # 如果已经初始化过，就直接跳过
        if self.superyolo_model is not None:
            print("SuperYOLO 已经初始化过，跳过。")
            return

        # 确保路径和 config 在 init 时已传进来
    # 先校验配置是否齐全
        if SuperYOLOModel is None:
            raise RuntimeError("SuperYOLOModel 类未导入，无法初始化。")
        if not self.yolo_cfg_path or not self.yolo_weights_path:
            raise RuntimeError("yolo_cfg_path/yolo_weights_path 在 config 中未配置，无法初始化。")
        if self.yolo_feature_layer_idx is None or self.yolo_feature_layer_channels is None:
            raise RuntimeError("yolo_feature_layer_idx/yolo_feature_layer_channels 在 config 中未配置，无法初始化。")
            # 此时配置齐全，开始加载
        device = torch.device(map_location)
        self.superyolo_model = SuperYOLOModel(
                        cfg=self.yolo_cfg_path,
                        ch=64, # Standard image channels
                        nc=getattr(self.config, 'nc', 1), # Number of classes, might not be needed for backbone only
                        input_mode='RGB+IR+MF',  # 显式设置 input_mode
                        anchors=getattr(self.config, 'anchors', None), # Anchors, might not be needed
                        sr=getattr(self.config, 'sr', False) # Set SR status as needed, potentially False if only using backbone
                    )

                # Load weights
        ckpt = torch.load(self.yolo_weights_path, map_location='cpu')
        if 'model' in ckpt:
            model = ckpt['model']  # 这是一个完整的模型对象
            yolo_weights = model.state_dict()  # 从模型实例中提取权重字典
        else:
            raise ValueError("Checkpoint does not contain a 'model' key")
        try:
            self.superyolo_model.load_state_dict(yolo_weights, strict=False) # strict=False allows ignoring missing/unexpected keys
            print(f"Loaded SuperYOLO weights from {self.yolo_weights_path} (strict=False).")
        except RuntimeError as e:
            print(f"RuntimeError loading SuperYOLO state_dict: {e}")
            print("Attempting to load with strict=True might fail if model architectures differ.")
                     # Potentially re-attempt with strict=True if expecting exact match, but strict=False is safer for partial loading.
        self.superyolo_model.eval()  # 切换到评估模式
        for param in self.superyolo_model.parameters():
            param.requires_grad = False
        print("SuperYOLO parameters are frozen.")
        if SuperYOLOModel is None:
            print("DEBUG: Reason: SuperYOLOModel class was not imported successfully.")
        if not self.yolo_cfg_path:
            print("DEBUG: Reason: yolo_cfg_path is missing or empty in LLaVA config.")
        if not self.yolo_weights_path:
            print("DEBUG: Reason: yolo_weights_path is missing or empty in LLaVA config.")
        if self.yolo_feature_layer_idx is None:
            print("DEBUG: Reason: yolo_feature_layer_idx is missing in LLaVA config.")
        if self.yolo_feature_layer_channels is None:
            print("DEBUG: Reason: yolo_feature_layer_channels is missing in LLaVA config.")
            print("SuperYOLO configuration incomplete or SuperYOLOModel not imported. SuperYOLO features will not be used.")
        if self.superyolo_model is not None and self.yolo_feature_layer_channels is not None:
            # Calculate the flattened dimension for each patch from the selected layer
            # flattened_patch_dim = channels * patch_size * patch_size
            flattened_dim = self.yolo_feature_layer_channels * self.yolo_patch_size * self.yolo_patch_size
            lm_hidden_size = self.config.text_config.hidden_size # Target dimension for projection
            intermediate_dim = min(flattened_dim, lm_hidden_size * 4) 
            try:
                self.yolo_projector = nn.Linear(flattened_dim, lm_hidden_size)
                # self.yolo_projector = nn.Sequential(
                #     nn.Linear(flattened_dim, intermediate_dim),
                #     nn.LayerNorm(intermediate_dim),
                #     nn.GELU(),
                #     nn.Linear(intermediate_dim, lm_hidden_size)
                #         )
                # # 初始化权重
                # for layer in self.yolo_projector:
                #     if isinstance(layer, nn.Linear):
                #         nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='gelu')  # 适用于 GELU/ReLU

                # 可选：初始化投影层权重（例如使用 kaiming_uniform_）
                # nn.init.kaiming_uniform_(self.yolo_projector.weight, a=math.sqrt(5))
                # if self.yolo_projector.bias is not None:
                #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.yolo_projector.weight)
                #     bound = 1 / math.sqrt(fan_in)
                #     nn.init.uniform_(self.yolo_projector.bias, -bound, bound)

                print(f"Defined YOLO projector for layer {self.yolo_feature_layer_idx} (channels {self.yolo_feature_layer_channels}) with input dim {flattened_dim}.")
            except Exception as e:
                print(f"Error defining YOLO projector: {e}")
                self.yolo_projector = None # Ensure None if definition fails
    def extract_superyolo_features(self, image_tensor: torch.Tensor) -> torch.Tensor | None:

        # print(f"DEBUG: extract_superyolo_features called with image_tensor shape: {image_tensor.shape}")
        if self.superyolo_model is None or self.yolo_projector is None or self.yolo_feature_layer_idx is None or self.yolo_feature_layer_channels is None:
            print("Warning: SuperYOLO components not fully initialized. Skipping YOLO feature extraction.")
            return None # Cannot process if model, projector, or config are missing
        # 将 image_tensor 转换为 torch.bfloat16
        image_tensor = image_tensor.to(torch.bfloat16)
        self.superyolo_model.to(image_tensor.device, dtype=image_tensor.dtype)
        self.yolo_projector.to(image_tensor.device, dtype=image_tensor.dtype)
        target_size = tuple(self.yolo_input_size) # (H_yolo, W_yolo), must be divisible by 32
        interpolated_image = image_tensor # Start with the original tensor
        if image_tensor.shape[2:] != target_size:
             interpolated_image = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)
        dummy_ir = torch.zeros(interpolated_image.shape[0], 1, target_size[0], target_size[1], device=interpolated_image.device)
        dummy_ir = dummy_ir.to(torch.bfloat16)
        # print(f"DEBUG: dummy_ir dtype: {dummy_ir.shape}")
        self.superyolo_model.eval()
        try:

             with torch.no_grad(): # Ensure no gradients are computed for SuperYOLO backbone if frozen
                 _, _, features_list = self.superyolo_model.forward(
                     interpolated_image, # Use interpolated image as the main input 'x'
                     dummy_ir,           # Pass dummy IR tensor
                     input_mode='RGB+IR+MF', # Specify input mode
                     augment=False,
                     profile=False,
                 )
             if self.yolo_feature_layer_idx < 0 or self.yolo_feature_layer_idx >= len(features_list):
                  print(f"Error: Configured YOLO feature layer index {self.yolo_feature_layer_idx} is out of bounds for SuperYOLO output list (length {len(features_list)}).")
                  return None

             feature_map = features_list[self.yolo_feature_layer_idx]
             # print(f"DEBUG: Extracted YOLO feature map from layer {self.yolo_feature_layer_idx} shape: {feature_map.shape}") # Expected: (B, 256, H_feat, W_feat)

             # Ensure the feature map has the expected channel dimension
             if feature_map.shape[1] != self.yolo_feature_layer_channels:
                  print(f"Error: Extracted YOLO feature map has {feature_map.shape[1]} channels, but expected {self.yolo_feature_layer_channels} for layer {self.yolo_feature_layer_idx}. Check yolo_feature_layer_channels config or SRyolo_MF.yaml/parse_model logic.")
                  return None

        except Exception as e:
             print(f"Error during SuperYOLO forward pass or feature extraction: {e}")
             return None

        B, C_feat, H_feat, W_feat = feature_map.shape
        patch_size = self.yolo_patch_size
        if H_feat % patch_size != 0 or W_feat % patch_size != 0:
            pad_h = (patch_size - (H_feat % patch_size)) % patch_size
            pad_w = (patch_size - (W_feat % patch_size)) % patch_size
            feature_map = F.pad(feature_map, (0, pad_w, 0, pad_h)) # Pad (left, right, top, bottom)
            # Update H_feat, W_feat after padding提取指定层特征并填充以适配切块。
            _, _, H_feat, W_feat = feature_map.shape
            # print(f"DEBUG: Feature map shape after padding: {feature_map.shape}")

        num_patches_h = H_feat // patch_size
        num_patches_w = W_feat // patch_size
        num_total_patches = num_patches_h * num_patches_w

        if num_total_patches <= 0:
             print(f"Warning: Zero or negative patches calculated for feature map. Skipping projection.")
             return None
        if feature_map is not None:
            pass
        patches = F.unfold(feature_map, kernel_size=patch_size, stride=patch_size) # Shape: [B, C*k*k, num_patches]
        patches_flattened = patches.transpose(1, 2) # Shape: [B, num_total_patches, C*k*k]
        # print(f"DEBUG: Flattened patches shape: {patches_flattened.shape}") # Expected: (B, num_total_patches, C_feat*patch_size*patch_size)
        # --- 观察投影前 YOLO 特征的分布 ---        
        if self.debug_features_viz:
            raw_yolo_flat = patches_flattened.reshape(-1) # All patches from all images in batch

        expected_flattened_dim = C_feat * patch_size * patch_size
        if patches_flattened.shape[-1] != expected_flattened_dim:
             print(f"Error: Flattened patch dimension ({patches_flattened.shape[-1]}) does not match expected projector input dimension ({expected_flattened_dim}). Check patch_size or yolo_feature_layer_channels.")
             return None

        # Project the flattened patches
        projected_patches = self.yolo_projector(patches_flattened) # Shape: [B, num_total_patches, hidden_size]
        logger.warning(f"YOLO 特征形状: {projected_patches.shape}")
        logger.warning(f"YOLO 特征 min: {projected_patches.min().item()}, max: {projected_patches.max().item()}, mean: {projected_patches.mean().item()}")
        # --- 观察投影后 YOLO 特征的分布 ---
        if self.debug_features_viz:
            projected_yolo_flat = projected_patches.reshape(-1) # All projected patches from all images in batch
            self._temp_raw_yolo_patches = patches_flattened.cpu().detach()
            self._temp_projected_yolo_patches = projected_patches.cpu().detach()
        return projected_patches

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, yolo_images=None
    ):
        print("DEBUG: Entering prepare_inputs_labels_for_multimodal...")
        print(f"pixel_values shape: {pixel_values.shape if pixel_values is not None else None}")
        print(f"yolo_images shape: {yolo_images.shape if yolo_images is not None else None}")
   
        image_token_id = self.config.image_token_id if hasattr(self.config, 'image_token_id') else 151646  # 根据实际情况调整
        if self.vision_tower is None or images is None:
            print("DEBUG: No vision tower, images, or input_ids shape[1]==1. Calling super method directly.")
            inputs_embeds = self.get_input_embeddings()(input_ids) 
            return (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels)
        if isinstance(images, torch.Tensor):
            # 处理标准图像批次 (B, C, H, W)
            print(f"训练时 CLIP 输入 (pixel_values) 形状: {images.shape}")
            print(f"训练时 CLIP 输入 范围: min={images.min().item()}, max={images.max().item()}")
            image_outputs = self.vision_tower(images, output_hidden_states=True)
            selected_image_feature = image_outputs.hidden_states[self.config.vision_feature_layer]
            print(f"训练时 Vision Tower 输出 形状: {selected_image_feature.shape}")
            print(f"训练时 Vision Tower 输出 范围: min={selected_image_feature.min().item()}, max={selected_image_feature.max().item()}")
        
            if self.config.vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            image_features_clip = self.multi_modal_projector(selected_image_feature)
            print(f"融合前 CLIP 特征维度: {image_features_clip.shape}")
            logger.warning(f"融合前 CLIP 特征样本: {image_features_clip[0, :5, :5]}")
            logger.warning(f"融合后特征(归一化前)2 min={image_features_clip.min()}, max={image_features_clip.max()}, mean={image_features_clip.mean()}")
            print(f"对齐后CLIP特征: min={image_features_clip.min().item():.4f}, max={image_features_clip.max().item():.4f}")
            # print(f"CLIP 特征维度: {image_features_clip.shape}")  # 检查 CLIP 特征
        elif isinstance(images, list):
            # 处理多图像输入（anyres 情况）
            print("Warning: Handling list of images for CLIP features is simplified. Processing each image separately.")
            image_features_list = []
            for img in images:
                if isinstance(img, torch.Tensor) and img.ndim == 4:
                    img_outputs = self.vision_tower(img, output_hidden_states=True)
                    selected_img_feature = img_outputs.hidden_states[self.config.vision_feature_layer]
                    if self.config.vision_feature_select_strategy == "default":
                        selected_img_feature = selected_img_feature[:, 1:]
                    img_features = self.multi_modal_projector(selected_img_feature)
                                # === 新增：特征对齐和裁剪 ===
                    image_features_list.append(img_features)
                else:
                    print(f"Warning: Skipping invalid image in list: {type(img)}")
            if image_features_list:
                image_features_clip = torch.cat(image_features_list, dim=1)  # 沿着 token 维度拼接
            else:
                image_features_clip = None
                print("Warning: No valid CLIP features extracted from image list.")
        else:
            print(f"Warning: Unsupported image type: {type(images)}. Skipping CLIP features.")
            image_features_clip = None                          
        superyolo_features = None
        if yolo_images is not None and yolo_images.ndim == 4: # Standard batch of images (B, C, H, W)
            print(f"原始 yolo_images 范围: min={yolo_images.min().item()}, max={yolo_images.max().item()}")
            superyolo_features = self.extract_superyolo_features(yolo_images)

        elif isinstance(yolo_images, list):
            print("Warning: Handling list of images for YOLO is not implemented.")
        else:
            print(f"Warning: Unexpected image input type/shape for YOLO: {type(yolo_images)}, hape={yolo_images.shape if isinstance(yolo_images, torch.Tensor) else 'N/A'}. Skipping YOLO features.")
            pass # superyolo_features remains None
        # --- 观察 CLIP 特征的分布 ---
        if self.debug_features_viz:
            clip_flat = image_features_clip.reshape(-1)

            self._temp_clip_features = image_features_clip.cpu().detach()

        if image_features_clip is not None and superyolo_features is not None:

            if image_features_clip.shape[0] == superyolo_features.shape[0] and image_features_clip.shape[2] == superyolo_features.shape[2]:
                logger.warning("特征处理流程统计:")
                print(f"1. CLIP原始: min={image_features_clip.min().item()}, max={image_features_clip.max().item()}")
                print(f"2. YOLO原始: min={superyolo_features.min().item()}, max={superyolo_features.max().item()}")
                combined_image_features = torch.cat([image_features_clip, superyolo_features], dim=1)
                logger.warning(f"融合后特征维度: {combined_image_features.shape}")  # 检查融合特征
                logger.warning(f"融合后特征样本: {combined_image_features[0, :5, :5]}")
                logger.warning(f"融合后特征(归一化前) min={combined_image_features.min()}, max={combined_image_features.max()}, mean={combined_image_features.mean()}")
            else:
                    print(f"Warning: CLIP ({image_features_clip.shape}) and SuperYOLO ({superyolo_features.shape}) feature shapes incompatible. Using CLIP only.")
                    combined_image_features = image_features_clip
        elif image_features_clip is not None:
            combined_image_features = image_features_clip
            print(f"Using CLIP only, shape: {combined_image_features.shape}")
            print(f"CLIP特征统计: min={image_features_clip.min().item()}, max={image_features_clip.max().item()}, mean={image_features_clip.mean().item()}")
        # 如果 SuperYOLO 特征未提取或不可用，则仅使用 CLIP 特征
        elif superyolo_features is not None:
            combined_image_features = superyolo_features
            print(f"Using SuperYOLO only, shape: {combined_image_features.shape}")
        else:
            print("Error: No image features extracted.")
            raise ValueError("No image features extracted.")
        
        # print(f"Combined image_features (before merging with text) shape: f"{combined_image_features.shape}")

        # 4. Use the parent's logic to merge combined features with text embeddings
        # We temporarily replace the parent's encode_images method to return our combined features.
        inputs_embeds = self.get_input_embeddings()(input_ids)
        # print(f"文本嵌入维度 (融合前): {inputs_embeds.shape}")
        if combined_image_features is not None:
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                combined_image_features, inputs_embeds, input_ids, attention_mask, labels
            )
            logger.debug(f"融合后输入嵌入维度: {inputs_embeds.shape}")
            

        # print("DEBUG: encode_images temporarily replaced.")
        # 临时替换 encode_images 方法，注入合并特征。
        try:
            # Get raw LLM text embeddings for comparison *before* vision features are injected
            # This requires access to the language model's embedding layer
            # LlavaForConditionalGeneration.model usually holds the LlamaModel/Qwen2Model etc.
            # And that model usually has an embed_tokens attribute.
            llm_text_embeds = self.language_model.model.embed_tokens(input_ids).cpu().detach()
            if self.debug_features_viz:
                llm_flat = llm_text_embeds.reshape(-1)

                self._temp_llm_text_embeds = llm_text_embeds

            if self.debug_features_viz and \
               hasattr(self, '_temp_projected_yolo_patches') and \
               hasattr(self, '_temp_clip_features') and \
               hasattr(self, '_temp_llm_text_embeds'): # Ensure all necessary features are collected
                
                # We also need the final combined embeddings for comparison
                self._temp_final_input_embeds = inputs_embeds.cpu().detach()
                
                # 可视化调试
                if self.debug_features_viz:

                        self._visualize_features_pca(
                            raw_yolo_patches=self._temp_raw_yolo_patches,
                            projected_yolo_patches=self._temp_projected_yolo_patches,
                            clip_features=self._temp_clip_features,
                            llm_text_embeds=self._temp_llm_text_embeds,
                            final_input_embeds=self._temp_final_input_embeds,
                            batch_idx=0, # Assuming we visualize the first sample in the batch
                            sample_id=self.visualization_counter

                        )
                        self.visualization_counter += 1
                # Clear temporary attributes after visualization
                del self._temp_raw_yolo_patches
                del self._temp_projected_yolo_patches
                del self._temp_clip_features
                del self._temp_llm_text_embeds
                del self._temp_final_input_embeds
        finally:
            # Restore the original encode_images method
            pass

        return (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels)
    
    def _visualize_features_pca(self, raw_yolo_patches, projected_yolo_patches, clip_features, llm_text_embeds, final_input_embeds, batch_idx=0, sample_id=0):
        if raw_yolo_patches is None or projected_yolo_patches is None or \
           clip_features is None or llm_text_embeds is None or final_input_embeds is None:
            print("Warning: Not all feature types available for PCA visualization.")
            return

        print(f"Starting PCA visualization for batch sample {sample_id}...")

        # Extract features for the specified batch index
        raw_yolo_sample = raw_yolo_patches[batch_idx].to(torch.float32).numpy() # (N_yolo_tokens, raw_dim)
        projected_yolo_sample = projected_yolo_patches[batch_idx].to(torch.float32).numpy() # (N_yolo_tokens, hidden_size)
        clip_sample = clip_features[batch_idx].to(torch.float32).numpy() # (N_clip_tokens, hidden_size)
        llm_sample = llm_text_embeds[batch_idx].to(torch.float32).numpy() # (N_text_tokens, hidden_size)
        final_input_sample = final_input_embeds[batch_idx].to(torch.float32).numpy() # (N_total_tokens, hidden_size)
        features_to_pca = []
        labels = []
        
        # Projected YOLO Features
        features_to_pca.append(projected_yolo_sample)
        labels.extend(['YOLO_proj'] * projected_yolo_sample.shape[0])

        # CLIP Features
        features_to_pca.append(clip_sample)
        labels.extend(['CLIP'] * clip_sample.shape[0])

        # LLM Text Embeddings (raw)
        features_to_pca.append(llm_sample)
        labels.extend(['LLM_text'] * llm_sample.shape[0])

        # Final Combined Input Embeddings (includes text and visual tokens)
        features_to_pca.append(final_input_sample)
        labels.extend(['Final_Input'] * final_input_sample.shape[0]) # This will be dense, likely covering others

        all_features_np = np.concatenate(features_to_pca, axis=0)

        # Perform PCA
        pca = PCA(n_components=2)
        transformed_features = pca.fit_transform(all_features_np)

        # Plotting
        plt.figure(figsize=(10, 8))
        unique_labels = list(set(labels))
        colors = plt.cm.get_cmap('Set1', len(unique_labels)) # Use a colormap
        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame({
            'PC1': transformed_features[:, 0],
            'PC2': transformed_features[:, 1],
            'Label': labels
        })
        fig = px.scatter(df, x='PC1', y='PC2', color='Label', title=f'PCA of Features (Sample {sample_id})')
        fig.write_html(os.path.join(self.debug_output_dir, f'feature_pca_sample_{sample_id}.html'))

        start_idx = 0
        for i, label_type in enumerate(['YOLO_proj', 'CLIP', 'LLM_text', 'Final_Input']):
            # Filter for each label type
            indices = [j for j, label in enumerate(labels) if label == label_type]
            if not indices:
                continue # Skip if no data for this label type

            # Extract data for this label type
            x = transformed_features[indices, 0]
            y = transformed_features[indices, 1]
            markers = {'YOLO_proj': 'o', 'CLIP': 'x', 'LLM_text': 's', 'Final_Input': '^'}
            plt.scatter(x, y, color=colors(i), label=label_type, alpha=0.3, s=6, marker=markers[label_type]) # s=10 for smaller points

        plt.title(f'PCA of Multi-modal Features (Sample {sample_id})')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.debug_output_dir, f'feature_pca_sample_{sample_id}.png'))
        plt.close() # Close plot to free memory

        print(f"PCA plot saved to {os.path.join(self.debug_output_dir, f'feature_pca_sample_{sample_id}.png')}")

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):

        return super().get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            **kwargs
        )

    def get_model(self):
         # LlavaForConditionalGeneration puts the language model inside a 'model' attribute
         return self.model # This should return the language model (e.g., LlamaModel, Qwen2Model)

if __name__ == "__main__":
    from run_show import ModelArguments, DataArguments, load_model_processor
    from show_llava.data import LlavaDataset, build_qaimage
    import torch
    import os
    import logging

    # 配置参数
    model_args = ModelArguments(model_name_or_path="/home/zty/LLaVA/show_model/model001", train_type="none")
    data_args = DataArguments(data_path="/home/zty/dataset")

    # 加载模型和处理器
    custom_model, processor = load_model_processor(model_args)
    config = custom_model.config
    config.debug_features_viz = True
    custom_model.to("cuda:0")
    custom_model.eval()

    # 加载数据集
    dataset = LlavaDataset(dataset_dir=data_args.data_path, file_name="detection1_qa.json", image_folder=os.path.join(data_args.data_path, "image"))
    if len(dataset) == 0:
        logging.error("Dataset is empty. Please check dataset path and file.")
        sys.exit(1)

    # 处理多个样本
    num_samples_to_process = 5  # 设置要处理的样本数
    for i, sample in enumerate(dataset):
        if i >= num_samples_to_process:
            break
        # logging.info(f"Processing sample {i}")
        human_input, chatbot_output, image_path, bboxes = sample

        # 处理图像和输入
        qa_image_output = build_qaimage(
            processor=processor,
            q_text=human_input,
            a_text=chatbot_output,
            image_path=image_path,
            bboxes=bboxes,
            noise_std=0.0,
            apply_noise=False
        )
        if qa_image_output is None:
            logging.error(f"Failed to process image for sample {i}")
            continue

        pixel_values = qa_image_output.pixel_values.to("cuda:0")
        input_token_ids = processor.tokenizer(human_input, return_tensors="pt", add_special_tokens=True).input_ids.to("cuda:0")
        def check_requires_grad(model):
            for name, param in model.named_parameters():
                if 'yolo_projector' in name:
                    print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

        # 在模型初始化后调用此函数
        check_requires_grad(custom_model)

        # 调试输入
        print(f"Sample {i} - Pixel values shape: {pixel_values.shape}, device: {pixel_values.device}")
        print(f"Sample {i} - Input token IDs shape: {input_token_ids.shape}, device: {input_token_ids.device}")

        # 生成输出并触发 PCA 可视化
        with torch.no_grad():
            output_ids = custom_model.generate(
                input_ids=input_token_ids,
                pixel_values=pixel_values,
                attention_mask=input_token_ids.ne(processor.tokenizer.pad_token_id),
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_beams=1
            )
        generated_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logging.info(f"Sample {i}: Generated Text: {generated_text}")

