#!/bin/zsh
uv run pitchtip scan "Carlos Rodon:2025:15" "Garrett Crochet:2025:15" "Luis Severino:2018:18" "Yusei Kikuchi:2025:15" \
  "Logan Webb:2025:15" "Jay Jackson:2023" "Mason Miller:2026" "Ryan Helsley:2024" "Tyler Glasnow:2021" \
  "Clarke Schmidt:2025" "Robbie Ray:2025:15" "Dylan Cease:2025:15" "Kevin Gausman:2025:15" "Paul Skenes:2025:15" \
  "Luis Castillo:2025:15" "Tanner Bibee:2025:15" "Hunter Brown:2025:15" "Cristopher Sanchez:2025:15" \
  "Tarik Skubal:2025:15" "Framber Valdez:2025:15" --workers 8 2>&1 | tr '\r' '\n' | grep -E "features for|FAILED"
