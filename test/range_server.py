#!/usr/bin/env python3
"""Minimal HTTP server with HTTP Range (206) support, for serving the GGUF to wllama."""
import http.server, socketserver, os, sys, posixpath, mimetypes

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8766

class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)
    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Expose-Headers', 'Content-Length, Content-Range')
        super().end_headers()
    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        ctype = self.guess_type(path)
        fsize = os.path.getsize(path)
        range_hdr = self.headers.get('Range')
        if range_hdr and range_hdr.startswith('bytes='):
            s, e = range_hdr[6:].split('-', 1)
            start = int(s) if s else 0
            end = int(e) if e else fsize - 1
            if start > end or start >= fsize:
                self.send_error(416); return None
            end = min(end, fsize - 1)
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{fsize}')
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Content-Type', ctype)
            self.end_headers()
            return open(path, 'rb')  # caller seeks? SimpleHTTPRequestHandler doesn't seek — handle below
        # full file
        self.send_response(200)
        self.send_header('Content-Length', str(fsize))
        self.send_header('Content-Type', ctype)
        self.end_headers()
        return open(path, 'rb')

    def copyfile(self, source, outputfile):
        # honor the range offset (source is an open file handle)
        range_hdr = self.headers.get('Range')
        if range_hdr and range_hdr.startswith('bytes='):
            s = range_hdr[6:].split('-',1)[0]
            if s:
                source.seek(int(s))
        super().copyfile(source, outputfile)

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('127.0.0.1', PORT), RangeHandler) as httpd:
    print(f'range-capable server: http://127.0.0.1:{PORT}/  (root={ROOT})')
    httpd.serve_forever()
