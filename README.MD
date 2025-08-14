# Core converters used by the app
sudo apt install -y libreoffice poppler-utils

# Fonts (for crisp email previews + monospace options)
sudo apt install -y fonts-dejavu-core fonts-liberation fonts-noto-core fonts-noto-mono

# Optional: HEIC support (only if pillow-heif complains about missing libheif)
sudo apt install -y libheif1

## Automatic classification

The Streamlit interface includes an **Auto-classify all files (no manual review)** button.
When used, every file in the input folder is classified using the model's suggestions
and filed automatically. Decisions logged during this mode mark the reviewer as
`automatic` and the notes as `not verified` so they can be distinguished from
manually reviewed entries.

