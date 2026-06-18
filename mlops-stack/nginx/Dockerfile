FROM nginx:1.25-alpine
RUN apk add --no-cache openssl && \
    mkdir -p /etc/nginx/ssl && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
      -keyout /etc/nginx/ssl/server.key \
      -out    /etc/nginx/ssl/server.crt \
      -subj   "/CN=localhost/O=MLOps-Stack"
COPY nginx.conf /etc/nginx/nginx.conf
