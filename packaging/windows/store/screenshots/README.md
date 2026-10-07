# Store screenshots

Drop the screenshots to use in the Microsoft Store listing here.

## Requirements (Microsoft Store)

* Format: **PNG** (preferred) or JPEG.
* Minimum size: **1366 × 768**.
* Recommended: **1920 × 1080** (16:9), landscape.
* At least **1** screenshot is required; **4 or more** are recommended.
* No misleading content, no watermarks, no copyrighted material you do not own.

## Suggested shots

1. Lighttable with a collection and thumbnails.
2. Darkroom with the Tone & Texture module open.
3. The ACES 2.0 reference rendering / 3DCF module.
4. A finished edit (before/after if possible).

## Validate

```bash
packaging/windows/store/validate-screenshots.sh packaging/windows/store/screenshots
```

The script checks the format and dimensions with ImageMagick (`magick`/`identify`).
