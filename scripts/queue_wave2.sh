#!/bin/zsh
# Second wave: documented-tip postseasons + more seasons for the most tippable pitchers.
uv run pitchtip scan "Stephen Strasburg:2019:14" --game-type R --game-type D --game-type L --game-type W --workers 8 2>&1 | tr '\r' '\n' | grep -E "features for|FAILED"
uv run pitchtip scan "Mason Miller:2026" --game-type R --game-type F --game-type D --workers 8 2>&1 | tr '\r' '\n' | grep -E "features for|FAILED"
uv run pitchtip scan "Ryan Helsley:2024" "Tyler Glasnow:2021" "Tyler Glasnow:2024:20" "Clarke Schmidt:2025" \
  "Zack Wheeler:2024:20" "Corbin Burnes:2024:20" "Chris Sale:2024:20" "Seth Lugo:2024:20" \
  "Cole Ragans:2024:20" "Framber Valdez:2024:20" --workers 8 2>&1 | tr '\r' '\n' | grep -E "features for|FAILED"
