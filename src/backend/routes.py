from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from .hf_hosting import HuggingFaceHostingService
from .db_manager import DatabaseManager

router = APIRouter()

class HostModelRequest(BaseModel):
    model_url: str
    custom_name: Optional[str] = None

@router.post("/host-model")
async def host_model(request: HostModelRequest, user_id: str = Depends(get_current_user)):
    """
    Host a new Hugging Face model
    """
    try:
        # Initialize services
        db_manager = DatabaseManager()
        hf_service = HuggingFaceHostingService(db_manager)
        
        # Register the model
        result = hf_service.register_model(
            user_id=user_id,
            model_url=request.model_url,
            custom_name=request.custom_name
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
            
        return {
            "success": True,
            "message": result['message'],
            "model_id": result['hosted_model_id'],
            "status": result['status']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ... existing code ... 