#!/bin/sh
for i in {0..8}; do
    python clearsign_client.py --id $i &
done &
python clearsign_client.py --id 9