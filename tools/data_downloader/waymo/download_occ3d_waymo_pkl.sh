#!/bin/bash

links=(
    "https://drive.google.com/uc?id=1zQ_7ZuZ2sPOhmIMH0BRj3s2V1qlaCFf7"
    "https://drive.google.com/file/d/1GMJd4aGS89wmjiL1G4fhWI6nToHatzXb/view?usp=sharing"
    "https://drive.google.com/file/d/15f8kYdLea09j6VRXKkxsFkoIFcTo6VeG/view?usp=sharing"
    "https://drive.google.com/file/d/1d2SAU0D1IZXSL6WHUwdbf-SFB2-hyOp0/view?usp=sharing"
)

for link in "${links[@]}"; do
  gdown "$link" --fuzzy
done
