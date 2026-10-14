"""共享测试夹具。

Antigravity 后端可用模型集合是进程级缓存（src.converter.gemini_fix），
模型列表路由会写入它；逐用例清零可保证映射测试不受测试顺序影响。
"""

import pytest


@pytest.fixture(autouse=True)
def reset_antigravity_available_model_cache():
    from src.converter import gemini_fix

    gemini_fix.set_antigravity_available_model_ids(set())
    yield
    gemini_fix.set_antigravity_available_model_ids(set())
