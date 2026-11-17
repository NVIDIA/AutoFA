# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use verified original image bytes when they decode to the same rendered pixels."""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path


def source_image_delivery(
    rendered: dict, source_fields: dict | None,
) -> tuple[dict, dict | None]:
    from autofa.data.artifact_access import image_content
    from autofa.data.artifact_store import ensure_artifact

    ordinary = image_content(rendered)
    # Small images need no extra work. Other artifact pages keep their renderer.
    if (len(ordinary['data']) <= 1024 * 1024 or not source_fields
            or rendered.get('format') != 'image' or rendered.get('page') != 1):
        return ordinary, None
    source = ensure_artifact(source_fields, rendered['artifact_idx'])
    if (source.get('status') != 'available' or source.get('verified') is not True
            or source.get('sha256') != rendered.get('artifact_sha256')):
        return ordinary, None
    # Reuse the artifact cache's size enforcement; verify bytes again before delivery.
    raw = Path(source['path']).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != source['sha256']:
        raise ValueError('Verified source image changed before delivery')
    encoded = base64.b64encode(raw).decode('ascii')
    if len(encoded) >= len(ordinary['data']):
        return ordinary, None
    pixels = base64.b64decode(ordinary['data'], validate=True)
    if hashlib.sha256(pixels).hexdigest() != rendered.get('sha256'):
        raise ValueError('Rendered image changed before delivery')
    from PIL import Image, ImageChops

    with Image.open(io.BytesIO(raw)) as original, Image.open(io.BytesIO(pixels)) as image:
        mime = {'JPEG':'image/jpeg', 'PNG':'image/png', 'WEBP':'image/webp'}.get(original.format)
        if (mime is None or getattr(original, 'n_frames', 1) != 1
                or original.mode != 'RGB' or image.mode not in {'RGB', 'RGBA'}
                or original.size != image.size
                or original.getexif().get(274, 1) != 1
                or original.info.get('icc_profile') != image.info.get('icc_profile')
                or (image.mode == 'RGBA' and image.getchannel('A').getextrema() != (255,255))
                or ImageChops.difference(original, image.convert('RGB')).getbbox() is not None):
            return ordinary, None
        width, height = image.size
        rendered_mode = image.mode
    delivery = {
        'kind': 'verified_source_image', 'sha256': digest,
        'rendered_sha256': rendered['sha256'], 'mime_type': mime,
        'width': width, 'height': height, 'decoded_rgb_identical': True,
        'rendered_mode': rendered_mode, 'rendered_alpha_opaque': True,
        'source_bytes': len(raw), 'rendered_bytes': len(pixels),
        'transform': 'Original bytes; no resize, crop, re-encoding or pixel changes.',
    }
    return {'type':'image', 'mimeType':mime, 'data':encoded}, delivery
