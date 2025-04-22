FROM alpine:latest
RUN apk add --no-cache curl
COPY ionos-updater.sh /ionos-updater.sh
RUN chmod +x /ionos-updater.sh
CMD ["sh", "-c", "while true; do /ionos-updater.sh; sleep 43200; done"]
