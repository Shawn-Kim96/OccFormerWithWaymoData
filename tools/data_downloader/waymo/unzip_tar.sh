cd data/waymo_v1-3-1/waymo_format

find . -type f -name "*.tar" | while read f; do
    echo "Extracting: $f"
    dir=$(dirname "$f")
    tar -xf "$f" -C "$dir"
done