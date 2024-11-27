from subprocess import run
from urllib.request import urlopen
from urllib.parse import urlencode
from time import sleep

api_dev_key = '***'

data = urlencode({
  'api_dev_key': api_dev_key,
  'api_user_name': 'cristhiangz',
  'api_user_password': '***',
}).encode('ascii')

with urlopen('https://pastebin.com/api/api_login.php', data) as res:
    api_user_key = res.read().decode()

def api_post(api_user_key):
    cp = run(('curl', '-s', 'ip.me'), capture_output=True)
    ip = cp.stdout.decode()

    data = urlencode({
      'api_dev_key': api_dev_key,
      'api_paste_code': ip,
      'api_option': 'paste',
      'api_paste_private': '2',
      'api_paste_name': 'gcca-yip_center',
      'api_paste_expire_date': '10M',
      'api_paste_fomat': 'txt',
      'api_user_key': api_user_key,
    }).encode('ascii')

    with urlopen('https://pastebin.com/api/api_post.php', data) as res:
        print(res.read().decode())

api_post(api_user_key)
while True:
    sleep(10 * 60)
    api_post(api_user_key)
