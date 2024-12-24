from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import ssl
import json
import urllib.parse
import gzip
import zlib
import warnings
import httpx

class LLMProxyHandler(BaseHTTPRequestHandler):
    save_dir = "recordings"
    request_counter = 0
    protocol_version = 'HTTP/1.1'  # Explicitly set HTTP version
    
    def decode_content(self, content, encoding):
        try:
            if encoding == 'gzip':
                return gzip.decompress(content)
            elif encoding == 'deflate':
                return zlib.decompress(content)
            else:
                warnings.warn(f"Unknown encoding: {encoding}")
            return content
        except Exception as e:
            warnings.warn(f"Failed to decode content: {e}")
            return content

    def do_POST(self):
        path_parts = self.path.split('/', 2)
        if len(path_parts) < 3:
            self.send_error(400, "Invalid URL format")
            return
        
        original_base_url = urllib.parse.unquote(path_parts[1])
        actual_path = '/' + path_parts[2]
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        current_count = LLMProxyHandler.request_counter
        LLMProxyHandler.request_counter += 1
        
        # Save request headers
        headers = dict(self.headers)
        request_headers_file = os.path.join(self.save_dir, f"{current_count}_request_headers.json")
        with open(request_headers_file, 'w') as f:
            json.dump(headers, f, indent=2)
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        # Save request body as-is
        request_body_file = os.path.join(self.save_dir, f"{current_count}_request_body.json")
        with open(request_body_file, 'wb') as f:
            f.write(body)
        
        # Forward request using httpx
        url = f"{original_base_url}{actual_path}"
        headers = dict(self.headers)
        headers.pop('Host', None)
        
        try:
            with httpx.Client(verify=True) as client:
                response = client.post(
                    url,
                    content=body,
                    headers=headers,
                )
                
                # Save response body
                response_data = response.content
                response_body_file = os.path.join(self.save_dir, f"{current_count}_response_body.json")
                with open(response_body_file, 'wb') as f:
                    f.write(response_data)
                
                # Send response to client
                self.send_response(response.status_code)
                
                # Set Content-Length header to ensure proper connection handling
                self.send_header('Content-Length', str(len(response_data)))
                
                # Copy other headers from the response
                for key, value in response.headers.items():
                    if key.lower() != 'content-length':  # Skip original Content-Length
                        self.send_header(key, value)
                
                # Ensure connection is closed after response
                self.send_header('Connection', 'close')
                self.end_headers()
                
                # Send response body
                self.wfile.write(response_data)
                self.wfile.flush()
                
        except httpx.RequestError as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Connection', 'close')
            error_response = json.dumps({
                'error': str(e),
                'detail': 'Failed to forward request'
            }).encode('utf-8')
            self.send_header('Content-Length', str(len(error_response)))
            self.end_headers()
            self.wfile.write(error_response)
            self.wfile.flush()

def run_proxy(port=8000, save_dir="recordings"):
    if os.path.exists(save_dir):
        for file in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, file))
    
    LLMProxyHandler.save_dir = save_dir
    server = HTTPServer(('localhost', port), LLMProxyHandler)
    print(f'Starting proxy server on port {port}...')
    print(f'Saving recordings to {save_dir}')
    server.serve_forever()

if __name__ == '__main__':
    run_proxy()