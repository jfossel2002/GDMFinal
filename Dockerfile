#Docker file for the envoy proxy that is broken
FROM envoyproxy/envoy:v1.18.3

COPY envoy.yaml /etc/envoy/envoy.yaml

CMD ["envoy", "-c", "/etc/envoy/envoy.yaml"]
