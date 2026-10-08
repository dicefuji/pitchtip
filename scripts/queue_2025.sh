#!/bin/zsh
# Waits for any running scan, then scans the 2025 pitch-count leaders and embeds them.
while pgrep -f "pitchtip scan" >/dev/null; do sleep 20; done
P=("Logan Webb" "Carlos Rodon" "Garrett Crochet" "Zac Gallen" "Max Fried" "Yusei Kikuchi"
   "Freddy Peralta" "Robbie Ray" "Dylan Cease" "Kevin Gausman" "Paul Skenes" "Luis Castillo"
   "Tanner Bibee" "Hunter Brown" "Gavin Williams" "Cristopher Sanchez" "Sandy Alcantara"
   "Framber Valdez" "Tarik Skubal" "Jack Flaherty" "Joe Ryan" "Chris Bassitt" "MacKenzie Gore" "Clay Holmes")
for p in $P; do
  uv run pitchtip scan "$p:2025:15" --workers 10 2>&1 | tr '\r' '\n' | grep -E "features for|Error|Traceback"
done
