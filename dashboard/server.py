"""Start Streamlit with the Access reconnect guard ahead of its frontend bundle."""
from pathlib import Path

import streamlit
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware import Middleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_INDEX = (Path(streamlit.__file__).with_name("static") / "index.html").read_bytes()
_MODULE = b'<script type="module"'
if _MODULE not in _INDEX:
    raise RuntimeError("Streamlit index has no module entry point for the reconnect guard")
_GUARD = Path(__file__).with_name("reconnect.js").read_bytes()
_GUARDED_INDEX = _INDEX.replace(_MODULE, b"<script>\n" + _GUARD + b"</script>\n    " + _MODULE, 1)


class _DashboardShell:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "GET":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if "range" in headers or (
            scope["path"] not in {"/", "/index.html"} and "text/html" not in headers.get("accept", "")
        ):
            await self.app(scope, receive, send)
            return

        # Let Streamlit resolve the page and keep its asset, API and socket routes.
        # Fetch the shell uncompressed and unconditionally, including on the first
        # deploy when a browser might still have validators for the unguarded shell.
        scope = {
            **scope,
            "headers": [
                (key, value) for key, value in scope["headers"]
                if key not in {b"accept-encoding", b"if-none-match", b"if-modified-since"}
            ],
            "extensions": {key: value for key, value in scope.get("extensions", {}).items()
                           if key != "http.response.pathsend"},
        }
        start = None
        body = bytearray()

        async def send_shell(message: Message) -> None:
            nonlocal start
            if message["type"] == "http.response.start" and message["status"] == 200:
                response_headers = Headers(raw=message["headers"])
                if (response_headers.get("content-type", "").startswith("text/html")
                        and response_headers.get("content-length") == str(len(_INDEX))):
                    start = message
                    return
            if start is not None and message["type"] == "http.response.body":
                body.extend(message.get("body", b""))
                if message.get("more_body"):
                    return
                content = bytes(body)
                if content == _INDEX:
                    content = _GUARDED_INDEX
                    response_headers = MutableHeaders(scope=start)
                    response_headers["content-length"] = str(len(content))
                    response_headers["cache-control"] = "no-store"
                    for key in ("etag", "last-modified", "accept-ranges"):
                        del response_headers[key]
                await send(start)
                await send({"type": "http.response.body", "body": content})
                return
            await send(message)

        await self.app(scope, receive, send_shell)


app = streamlit.App(Path(__file__).with_name("app.py"), middleware=[Middleware(_DashboardShell)])
