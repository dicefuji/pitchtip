#!/bin/zsh
# Single prioritized YOLO11x lane (GPU-bound: one lane is as fast as two and avoids MPS contention).
uv run pitchtip extract "Yu Darvish 2017" 2>&1 | tr '\r' '\n' | grep "features for"; uv run pitchtip embed "Yu Darvish 2017" 2>&1 | grep ": ("
uv run pitchtip scan "Ryan Helsley:2025" "Freddy Peralta:2025:15" "Max Fried:2025:15" "Carlos Rodon:2025:15" \
  "Jesus Luzardo:2025" "Zac Gallen:2025:15" "Garrett Crochet:2025:15" "Max Scherzer:2025:17" --workers 6 2>&1 | tr '\r' '\n' | grep -E "features for|FAILED"
for k in "Tyler Glasnow 2020" "Clarke Schmidt 2024"; do
  uv run pitchtip extract "$k" 2>&1 | tr '\r' '\n' | grep "features for"; uv run pitchtip embed "$k" 2>&1 | grep ": ("
done
uv run pitchtip scan "Luis Severino:2018:18" "Yoshinobu Yamamoto:2024:18" "Mason Miller:2026" "Stephen Strasburg:2019:14" \
  "Hunter Greene:2025:20" "Yusei Kikuchi:2025:15" "Logan Webb:2025:15" "Jay Jackson:2023" "Ryan Helsley:2024" \
  "Tyler Glasnow:2021" "Clarke Schmidt:2025" "Robbie Ray:2025:15" "Dylan Cease:2025:15" "Kevin Gausman:2025:15" \
  "Paul Skenes:2025:15" "Luis Castillo:2025:15" "Tanner Bibee:2025:15" "Hunter Brown:2025:15" \
  "Cristopher Sanchez:2025:15" "Tarik Skubal:2025:15" "Framber Valdez:2025:15" --workers 6 2>&1 | tr '\r' '\n' | grep -E "features for|FAILED"
