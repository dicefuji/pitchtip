#!/bin/zsh
for k in "Yu Darvish 2017" "Tyler Glasnow 2020" "Clarke Schmidt 2024"; do
  uv run pitchtip extract "$k" 2>&1 | tr '\r' '\n' | grep -E "features for"; uv run pitchtip embed "$k" 2>&1 | grep -E ": \("
done
uv run pitchtip scan "Ryan Helsley:2025" "Freddy Peralta:2025:15" "Max Fried:2025:15" "Jesus Luzardo:2025" \
  "Zac Gallen:2025:15" "Max Scherzer:2025:17" "Yoshinobu Yamamoto:2024:18" "Hunter Greene:2025:20" \
  "Stephen Strasburg:2019:14" --workers 8 2>&1 | tr '\r' '\n' | grep -E "features for|FAILED"
