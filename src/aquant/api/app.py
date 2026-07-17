from collections.abc import Callable

from fastapi import FastAPI

from aquant.monitoring import LiveReadiness


def create_app(
    readiness_provider: Callable[[], LiveReadiness] | None = None,
) -> FastAPI:
    app = FastAPI(title="AQuant API", version="0.1.0", docs_url="/docs")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "service": "aquant"}

    @app.get("/ready")
    def readiness() -> dict[str, object]:
        state = (
            readiness_provider()
            if readiness_provider is not None
            else LiveReadiness(False, ("READINESS_PROVIDER_NOT_CONFIGURED",))
        )
        return {"ready": state.ready, "blockers": state.blockers}

    @app.get("/version")
    def version() -> dict[str, str]:
        return {"version": "0.1.0", "live_default": "disabled"}

    return app
