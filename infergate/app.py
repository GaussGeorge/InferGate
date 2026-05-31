from __future__ import annotations

from .proxy import ProxyConfig, create_app

app = create_app(ProxyConfig.from_env())


def main() -> None:
    import uvicorn

    config = ProxyConfig.from_env()
    uvicorn.run("infergate.app:app", host=config.host, port=config.port, reload=False)


if __name__ == "__main__":
    main()

