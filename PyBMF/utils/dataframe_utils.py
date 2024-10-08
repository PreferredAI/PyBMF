import os
import pandas as pd
import webbrowser
import re
import base64
from .download_utils import get_cache_path


# def get_config(key):
#     '''Get config value from settings.ini.

#     Parameters
#     ----------
#     key : str
#         Key in settings.ini.

#     Returns
#     -------
#     value : str
#         Value in settings.ini.
#     '''
#     has_config = os.path.isfile('settings.ini')

#     if has_config:
#         import configparser
#         config = configparser.ConfigParser()
#         config_path = os.path.abspath('settings.ini')
#         print("[I] Found settings.ini at", config_path)
#         config.read(config_path)
#         path= config["PATHS"][key]
#     else:
#         print("[W] No settings.ini found.")
#         path = None
#     return path


def get_chrome_path():
    '''Get the command of launching a browser. Google Chrome, specifically.
    '''
    import platform

    command = None

    if platform.system() == 'Windows':
        if os.path.exists('C:/Program Files/Google/Chrome/Application/chrome.exe'):
            command = r'C:/Program Files/Google/Chrome/Application/chrome.exe'
    elif platform.system() == 'Darwin':
        if os.path.exists('/Applications/Google Chrome.app'):
            # command = r'open -a /Applications/Google\ Chrome.app'
            command = r'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    elif platform.system() == 'Linux':
        if os.path.exists('/usr/bin/google-chrome'):
            command = r'/usr/bin/google-chrome'
    else:
        raise ValueError('Unknown OS type.')
    
    if command is None:
        raise ValueError('Google Chrome not found. Specify your path manually.')
    
    return command



def log2html(df, log_name, open_browser=True, log_path=None, browser_path=None):
    '''Display and save a dataframe in HTML, and open it in browser if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to be displayed in HTML.
    log_name : str
        Name of the log file.
    open_browser : bool
        Whether to open in browser.
    log_path : str
        Path to save the log file.
    browser_path : str
        Path of the browser.
    '''
    html_head = '''<!DOCTYPE html>
<html>
<head>
  <style>
    table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
        font-family: math;
    }
    th, td {
        border-style: dotted;
        border-color: black;
    }
    table.dataframe td:nth-child(odd) {
        background-color: #cecece;
    }

    table.dataframe tr:nth-child(even) {
        background-color: #e8e8e8;
    }
  </style>
</head>
<body>
'''

    html_tail = '''
</body>
'''

    html = df.to_html()
    html = html_head + html + html_tail

    full_path = _make_html(log_path, log_name, html)
    if open_browser:
        _open_html(full_path, browser_path)


def log2latex(df, log_name, open_browser=True, log_path=None, browser_path=None):
    '''Display a dataframe in TeX on overleaf.com.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to be displayed in TeX.
    log_name : str
        Name of the log file.
    open_browser : bool
        Whether to open in browser.
    log_path : str
        Path to save the log file.
    browser_path : str
        Path of the browser.
    '''
    width = int(df.columns.size * 0.8)
    height = int(len(df) * 0.2) + 1.0

    geometry = f"[left=20px,right=10px,top=10px,bottom=20px,paperwidth={width}in,paperheight={height}in]"

    latex_head = r'''\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{lscape}
\usepackage[table]{xcolor}
\usepackage''' + geometry + r'''{geometry}
\title{''' + log_name + r'''}
\begin{document}
'''

    latex_tail = r'''
\end{document}
'''

    latex = df.to_latex()

    # escape underscore to display properly in LaTeX
    latex = latex.replace('_', r'\_')

    latex = latex_head + latex + latex_tail

    html_head = '''
<body onload="document.forms['open_overleaf'].submit()">
  <form action="https://www.overleaf.com/docs" method="post" name="open_overleaf">
    <input type="text" name="snip_uri" value="data:application/x-tex;base64,'''
    
    html_tail = '''"><br>
  </form>
</body>
'''

    latex_bytes = latex.encode("ascii")
    latex_b64code = base64.b64encode(latex_bytes) 
    latex_b64str = latex_b64code.decode("ascii")
    html = html_head + latex_b64str + html_tail
    
    full_path = _make_html(log_path, log_name + " overleaf", html)
    if open_browser:
        _open_html(full_path, browser_path)


def _make_name(model=None, model_name=None, format="%Y-%m-%d %H-%M-%S-%f "):
    '''Make a file name for an instance of a model.

    Milliseconds are added to the end of the name to make it unique.

    Parameters
    ----------
    model : object
        Model object.
    model_name : str
        Name of the model.
    format : str
        Format of the timestamp.

    Returns
    -------
    model_name : str
        Name of the model.
    '''
    if model is None and model_name is None:
        print("[E] In _make_name(), model and model_name cannot be both None.")

    if model_name is None:
        model_name = str(type(model))
        model_name = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', model_name)[-3]

    model_name = pd.Timestamp.now().strftime(format) + model_name

    return model_name


def _make_html(file_path, file_name, html):
    '''Make a html file.

    Parameters
    ----------
    file_path : str
        Path to save the html file.
    file_name : str
        Name of the html file.
    html : str
        HTML code.

    Returns
    -------
    full_path : str
        Full path of the html file.
    '''
    if file_path is not None:
        full_path = os.path.join(os.path.abspath(file_path), file_name + ".html")
    else:
        full_path, _ = get_cache_path(relative_path='saved_logs/' + file_name + '.html')

    with open(full_path, "w") as f:
        f.write(html)

    print("[I] HTML saved as: " + os.path.abspath(full_path))

    return full_path


def _open_html(full_path, browser_path):
    '''Open a html file in browser.

    Parameters
    ----------
    full_path : str
        Full path of the html file.
    browser_path : str
        Path of the browser.
    '''
    if browser_path is not None:
        browser_path = browser_path + r' %s'
    else:
        browser_path = get_chrome_path() + r' %s'
    
    url = 'file:///' + os.path.abspath(full_path)

    print("[I] Opening HTML in browser: " + browser_path)

    webbrowser.get(using=browser_path).open(url, new=2)
