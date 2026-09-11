FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY static/ /usr/share/nginx/html/static/
COPY static/index.html /usr/share/nginx/html/index.html

EXPOSE 80
