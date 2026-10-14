"""
Antigravity Model List Router - Handles model list requests
Antigravity 模型列表路由 - 处理模型列表请求
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 第三方库
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

# 本地模块 - 工具和认证
from src.utils import (
    get_base_model_from_feature_model,
    authenticate_flexible
)

# 本地模块 - API
from src.api.antigravity import fetch_available_models
from src.converter.gemini_fix import set_antigravity_available_model_ids

# 后端仍会广告、但不应出现在对外列表中的 ID：
# chat_* 对聊天请求返回 400 INVALID_ARGUMENT；
# 3.5 Flash 全系与 3-flash-agent/preview 返回“Gemini 3.5 Flash is no longer available.”
# 停用通知；gemini-4-argon-disabled 实测 404（后端侧禁用）；均为 2026-10-14 实测。
_ANTIGRAVITY_HIDDEN_MODEL_IDS = frozenset({
    "gemini-4-argon-disabled",
    "chat_20706",
    "chat_23310",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash-low",
    "gemini-3.5-flash-extra-low",
    "gemini-3-flash-agent",
    "gemini-3-flash-preview",
})

# 后端只广告 tiered 档位的模型：为裸名与 high/medium/low 生成别名，
# 请求时由 gemini_fix 统一收敛到 tiered；后端开放固定档位后自动直连。
_ANTIGRAVITY_TIERED_FAMILIES = ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash")


# 本地模块 - 基础路由工具
from src.router.base_router import create_gemini_model_list, create_openai_model_list
from src.models import model_to_dict
from log import log


# ==================== 路由器初始化 ====================

router = APIRouter()


# ==================== 辅助函数 ====================

async def get_antigravity_models_with_features():
    """
    获取 Antigravity 模型列表并添加功能前缀
    
    Returns:
        带有功能前缀的模型列表
    """
    # 从 API 获取基础模型列表
    base_models_data = await fetch_available_models()
    
    if not base_models_data:
        log.warning("[ANTIGRAVITY MODEL LIST] 无法获取模型列表，返回空列表")
        return []
    
    # 提取模型 ID
    base_model_ids = [model['id'] for model in base_models_data if 'id' in model]

    # 刷新映射层缓存，供请求映射做同族降级
    set_antigravity_available_model_ids(base_model_ids)

    # 对外列表排除内部/已停用 ID
    visible_model_ids = [
        m for m in base_model_ids if m not in _ANTIGRAVITY_HIDDEN_MODEL_IDS
    ]

    # 添加功能前缀
    models = []
    for base_model in visible_model_ids:
        # 基础模型
        models.append(base_model)
        
        # 假流式模型 (前缀格式)
        models.append(f"假流式/{base_model}")
        
        # 流式抗截断模型 (仅在流式传输时有效，前缀格式)
        models.append(f"流式抗截断/{base_model}")

    # tiered-only 家族：补齐裸名与固定档位别名（映射层收敛到 tiered）
    for tiered_family in _ANTIGRAVITY_TIERED_FAMILIES:
        if f"{tiered_family}-tiered" not in base_model_ids:
            continue
        for alias in (tiered_family, f"{tiered_family}-high",
                      f"{tiered_family}-medium", f"{tiered_family}-low"):
            if alias in models:
                continue
            models.append(alias)
            models.append(f"假流式/{alias}")
            models.append(f"流式抗截断/{alias}")

    log.info(f"[ANTIGRAVITY MODEL LIST] 生成了 {len(models)} 个模型（包含功能前缀）")
    return models



# ==================== API 路由 ====================

@router.get("/antigravity/v1beta/models")
async def list_gemini_models(token: str = Depends(authenticate_flexible)):
    """
    返回 Gemini 格式的模型列表
    
    从 src.api.antigravity.fetch_available_models 动态获取模型列表
    并添加假流式和流式抗截断前缀
    """
    models = await get_antigravity_models_with_features()
    log.info("[ANTIGRAVITY MODEL LIST] 返回 Gemini 格式")
    return JSONResponse(content=create_gemini_model_list(
        models,
        base_name_extractor=get_base_model_from_feature_model
    ))


@router.get("/antigravity/v1/models")
async def list_openai_models(token: str = Depends(authenticate_flexible)):
    """
    返回 OpenAI 格式的模型列表
    
    从 src.api.antigravity.fetch_available_models 动态获取模型列表
    并添加假流式和流式抗截断前缀
    """
    models = await get_antigravity_models_with_features()
    log.info("[ANTIGRAVITY MODEL LIST] 返回 OpenAI 格式")
    model_list = create_openai_model_list(models, owned_by="google")
    return JSONResponse(content={
        "object": "list",
        "data": [model_to_dict(model) for model in model_list.data]
    })
