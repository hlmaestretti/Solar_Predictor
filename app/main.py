from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings


def get_application() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    app.include_router(router)
    return app


app = get_application()
