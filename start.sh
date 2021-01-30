#!/bin/sh
for i in {0..8}; do
    python securesign_client.py --id $i &
done &
python securesign_client.py --id 9