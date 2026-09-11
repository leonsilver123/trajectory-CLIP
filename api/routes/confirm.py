"""
api.routes.confirm - 确认 API

提供用户确认目标接口:
- POST /api/v1/confirm/target  用户确认目标实例
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.common.session_store import (
    InvalidStateTransition,
    SessionNotFoundError,
    get_session_store,
)

router = APIRouter()


class ConfirmRequest(BaseModel):
    """确认请求"""
    query_id: str                       # 查询 ID
    instance_id: str                    # 确认的目标实例 ID


class ConfirmResponse(BaseModel):
    """确认响应"""
    query_id: str
    instance_id: str
    status: str                         # "confirmed"
    message: str


@router.post("/target", response_model=ConfirmResponse)
async def confirm_target(request: ConfirmRequest):
    """
    用户确认目标

    用户从候选图片中确认目标后，系统开始轨迹回溯。
    确认动作写入会话状态（searched → confirmed），
    query_id 必须是检索接口返回过的有效会话，否则返回 404。
    """
    store = get_session_store()
    try:
        store.confirm(request.query_id, request.instance_id)
    except SessionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"查询 {request.query_id} 不存在或已失效，请重新检索",
        )
    except InvalidStateTransition as e:
        raise HTTPException(status_code=409, detail=str(e))

    return ConfirmResponse(
        query_id=request.query_id,
        instance_id=request.instance_id,
        status="confirmed",
        message=f"目标 {request.instance_id} 已确认，可进行轨迹回溯",
    )
