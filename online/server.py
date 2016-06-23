import time
import BaseHTTPServer
from captcha import init_model, predict
import urlparse
import re
import StringIO

model = init_model()

class SimpleHTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def __init__(self, a, b, c):
        BaseHTTPServer.BaseHTTPRequestHandler.__init__(self,a,b,c)

    def _get_boundary(self):
        o = self.headers.getplist()
        for i in o:
            x = i.split('=')
            if x[0] == 'boundary' and x[1] != '':
                return x[1]
        return None

    def get_path(self):
        a = re.sub(r'\?.*', '', self.path)
        return a

    def deal_post_data(self, fr):
        boundary = self._get_boundary()
        if not boundary:
            return False
        remainbytes = int(self.headers['content-length'])
        line = self.rfile.readline()
        remainbytes -= len(line)
        if not boundary in line:
            return False

        state = 'meta'
        newstate = 'meta'
        fdata = ''
        files = []
        filenames = []

        while True:
            line = self.rfile.readline()
            if line.startswith('--%s--' % boundary):
                newstate = 'done'
            elif line.startswith(boundary):
                newstate = 'meta'
            elif line == '\r\n':
                if state == 'fmeta':
                    newstate = 'fdata'
                elif state == 'meta':
                    newstate = 'data'
            else:
                f = re.search(r'Content-Disposition.*name="(.*)"; filename="(.*)"', line)
                if f and state == 'meta':
                    filenames.append([f.group(1), f.group(2)])
                    newstate = 'fmeta'

            if state == 'fdata':
                if newstate == 'fdata':
                    fdata += line
                else:
                    if fdata[-2:] == '\r\n':
                        fdata = fdata[0:-2]
                    fds = StringIO.StringIO()
                    fds.write(fdata)
                    fds.seek(0)
                    files.append(fds)
                    fdata = ''
            if newstate == 'done':
                break
            state = newstate

        fr['filenames'] = filenames
        fr['files'] = files
        return True

    def do_GET(self):
        self.send_response(404)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write('not found\n')

    def do_POST(self):
        up=urlparse.urlparse(self.path)
        reqpath = up.path
        if reqpath == '/':
            return self.on_codeocr()
        else:
            self.do_HEAD()
            self.wfile.write('not found\n')

    def on_codeocr(self):
        print "start", time.time()
        fr = {}
        r = self.deal_post_data(fr)
        if len(fr.get('files', []))>0:
            img = fr['files'][0]
            resp = predict(model, img)
            length = len(resp)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(resp)
            print resp
        print "end", time.time()


from SocketServer import ThreadingMixIn

class MyServer(ThreadingMixIn, BaseHTTPServer.HTTPServer):
    def __init__(self, a, b):
        BaseHTTPServer.HTTPServer.__init__(self, a,b)
        #self.tempfile_mgr = TempFileNames()


def test():
    httpd = MyServer(('', 5000), SimpleHTTPRequestHandler)
    print time.asctime(), "Server Starts"
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print time.asctime(), "Server Stops"

if __name__ == '__main__':
    test()
