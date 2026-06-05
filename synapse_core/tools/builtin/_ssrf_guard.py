"""SSRF protection utility for HTTP-based tools."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse


def _is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    except ValueError:
        return True


def _resolve_and_check(host: str) -> None:
    """Raise ValueError if host resolves to a private/internal IP."""
    try:
        addr_infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname '{host}': {e}") from e

    for addr_info in addr_infos:
        ip_str = addr_info[4][0]
        if _is_private_ip(ip_str):
            raise ValueError(
                f"Requests to private/internal addresses are not allowed (resolved: {ip_str})"
            )


async def check_ssrf(url: str) -> None:
    """Validate that url is safe to request (no SSRF).

    Raises ValueError with a user-visible message if the URL is unsafe.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported URL scheme '{parsed.scheme}': only http and https are allowed"
        )

    host = parsed.hostname
    if not host:
        raise ValueError("URL has no hostname")

    # Run blocking DNS resolution off the event loop
    await asyncio.to_thread(_resolve_and_check, host)
