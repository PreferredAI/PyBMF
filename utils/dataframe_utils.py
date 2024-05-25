import os
import pandas as pd
import webbrowser
import re
import base64


browser_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'


def log2html(model, df_name, file_path=r'..\logs', file_name=None, browser_path=browser_path, open_browser=True):
    '''Display a dataframe in HTML.
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

    html = model.logs[df_name].to_html()
    html = html_head + html + html_tail

    file_name = _make_name(model, file_name)
    full_path = _make_html(file_path, file_name, html)
    if open_browser:
        _open_html(full_path, browser_path)


def log2latex(model, df_name, file_path=r'..\logs', file_name=None, browser_path=browser_path, open_browser=True):
    '''Display a dataframe in TeX on overleaf.com.

    This tool automatically highlights the maximum values in each column. 
    '''
    width = int(model.logs[df_name].columns.size * 0.8)
    height = int(len(model.logs[df_name]) * 0.2) + 1.0

    geometry = f"[left=20px,right=10px,top=10px,bottom=20px,paperwidth={width}in,paperheight={height}in]"
    file_name = _make_name(model, file_name)

    latex_head = r'''\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{lscape}
\usepackage[table]{xcolor}
\usepackage''' + geometry + r'''{geometry}
\title{''' + file_name + r'''}
\begin{document}
'''

    latex_tail = r'''
\end{document}
'''

    latex = model.logs[df_name].style.highlight_max(props='cellcolor:[HTML]{FFFF00}; color:{red};')
    latex = latex.to_latex(hrules=False, clines="skip-last;data", multicol_align='c')
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
    
    full_path = _make_html(file_path, file_name, html)
    if open_browser:
        _open_html(full_path, browser_path)


def _make_name(model, file_name=None, format="%y-%m-%d_%H-%M-%S_"):
    if file_name is None:
        file_name = str(type(model))
        file_name = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', file_name)[-3]
    file_name = pd.Timestamp.now().strftime(format) + file_name
    return file_name


def _make_html(file_path, file_name, html):
    full_path = os.path.join(os.path.abspath(file_path), file_name + ".html")
    with open(full_path, "w") as f:
        f.write(html)
    print("[I] HTML saved as: {}.html".format(file_name))
    return full_path


def _open_html(full_path, browser_path):
    webbrowser.get(using=browser_path).open('file:///' + full_path, new=2)
