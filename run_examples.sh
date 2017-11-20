#!/bin/sh

echo "Running the examples to produce HTML reports..."
find examples -name "*.md" | xargs -n 1 pweave -f md2html
