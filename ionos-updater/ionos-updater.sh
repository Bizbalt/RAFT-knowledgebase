#!/bin/sh

# Path to store the previous IP
IP_FILE="/data/last_ip"
UPDATE_URL_FILE="/data/ionos_update_url" # This file must be created manually

# Retrieve the current public IP
CURRENT_IP=$(curl -s https://api.ipify.org)
if [ -z "$CURRENT_IP" ]; then
    echo "Error: Could not retrieve the current IP." >&2
    exit 1
fi

# Read the previous IP, if it exists
if [ -f "$IP_FILE" ]; then
    LAST_IP=$(cat "$IP_FILE")
else
    LAST_IP=""
fi

# Read the update URL from the file
if [ -f "$UPDATE_URL_FILE" ]; then
    UPDATE_URL=$(cat "$UPDATE_URL_FILE")
else
    echo "Error: Update URL file not found." >&2
    exit 1
fi

# Check if the IP has changed
if [ "$CURRENT_IP" != "$LAST_IP" ]; then
    echo "IP changed: $LAST_IP -> $CURRENT_IP"
    # Update the IP
    if curl -s -X GET "$UPDATE_URL?ip=$CURRENT_IP"; then
        echo "$CURRENT_IP" > "$IP_FILE"
    else
        echo "Error: Could not update the IP." >&2
        exit 1
    fi
else
    echo "IP unchanged: $CURRENT_IP"
fi