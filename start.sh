#!/bin/sh
for i in {0..8}; do
    python clearmedian_client.py --id $i &
done &
python clearmedian_client.py --id 9